/*
 * GenCache.cpp
 *
 * 從 MongoDB 讀取 BTC key r 字串，提取最後 10 hex 字元 (40 bits) 作為查找 tag，
 * 產生本地二進位快取供 CalcR_GPU 使用。
 *
 * 工作流程（兩階段）：
 *   1. build-part  : 從 MongoDB 單一 collection 讀所有文件 → 排序去重 → 寫 part 檔
 *   2. merge-parts : k-way merge 多個排序 part 檔 → 產生最終 cache.bin
 *
 * 為何兩階段：
 *   整個 collection 可能有數十億筆，RAM 不夠一次 sort。
 *   先按 collection 分成 part，再做 k-way merge 就能以串流方式合併，RAM 開銷固定。
 *
 * 編譯：
 *   g++ -O2 -std=c++17 -o GenCache GenCache.cpp \
 *       $(pkg-config --cflags --libs libmongoc-1.0)
 *
 * 用法：
 *   ./GenCache build-part <collection> <part_file>
 *   ./GenCache merge-parts <output_cache_file> <part1> <part2> ...
 *
 * 檔案格式：
 *   Part  file : magic(8) "PARTA040" | count(uint64_t) | Tag72[]
 *   Cache file : magic(8) "BTCTA040" | count(uint64_t) | Tag72[]
 *
 * Tag72 (5 bytes, packed) :
 *   uint8_t  hi  — 最後 10 hex 的高 8 bits（chars 0-1 of last 10）
 *   uint32_t lo  — 最後 10 hex 的低 32 bits（chars 2-9 of last 10）
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <queue>
#include <string>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

// ===========================================================================
// 40-bit tag 結構 (packed — 5 bytes, 無 padding)
// last 10 hex chars of r:  hex[0..1] → hi (8-bit), hex[2..9] → lo (32-bit)
// ===========================================================================
struct Tag72 {
    uint8_t  hi;   // bits 32-39
    uint32_t lo;   // bits 0-31
} __attribute__((packed));

static inline bool tag72_less(const Tag72 &a, const Tag72 &b) {
    if (a.hi != b.hi) return a.hi < b.hi;
    return a.lo < b.lo;
}
static inline bool tag72_eq(const Tag72 &a, const Tag72 &b) {
    return a.hi == b.hi && a.lo == b.lo;
}

// ===========================================================================
// 從字串尾部解析最後 10 個 hex 字元 → Tag72
// ===========================================================================
static bool parse_last10hex(Tag72 *out, const char *s, unsigned int len) {
    if (len < 10) return false;
    const char *p = s + (len - 10);

    // hi: 2 hex chars → 8 bits
    uint8_t hi = 0;
    for (int i = 0; i < 2; i++) {
        char c = p[i];
        uint8_t nibble;
        if      (c >= '0' && c <= '9') nibble = (uint8_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (uint8_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (uint8_t)(c - 'A' + 10);
        else return false;
        hi = (uint8_t)((hi << 4) | nibble);
    }

    // lo: 8 hex chars → 32 bits
    uint32_t lo = 0;
    for (int i = 2; i < 10; i++) {
        char c = p[i];
        uint32_t nibble;
        if      (c >= '0' && c <= '9') nibble = (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (uint32_t)(c - 'A' + 10);
        else return false;
        lo = (lo << 4) | nibble;
    }

    out->hi = hi;
    out->lo = lo;
    return true;
}

// ===========================================================================
// 檔案格式 magic
// ===========================================================================
static const char PART_MAGIC[8]  = {'P','A','R','T','A','0','4','0'};
static const char CACHE_MAGIC[8] = {'B','T','C','T','A','0','4','0'};

// ===========================================================================
// MongoDB 連線設定（可透過環境變數 MONGO_URI 覆蓋）
// ===========================================================================
static const char *DEFAULT_MONGO_URI =
    "mongodb://192.168.50.173:27017/"
    "?directConnection=true"
    "&serverSelectionTimeoutMS=30000"
    "&connectTimeoutMS=30000"
    "&socketTimeoutMS=600000";

static const char *DB_NAME = "ecdsa";

// ===========================================================================
// 工具函式
// ===========================================================================
static void print_usage() {
    puts("Usage:");
    puts("  ./GenCache build-part <collection> <part_file>");
    puts("  ./GenCache merge-parts <output_cache_file> <part1> <part2> ...");
}

static bool save_part_file(const char *path, const std::vector<Tag72> &v) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;

    uint64_t count = (uint64_t)v.size();
    bool ok = (fwrite(PART_MAGIC, 1, 8, f) == 8) &&
              (fwrite(&count, sizeof(count), 1, f) == 1) &&
              (v.empty() || fwrite(v.data(), sizeof(Tag72), v.size(), f) == v.size());
    fclose(f);
    return ok;
}

// ===========================================================================
// MongoDB → vector<Tag72>
// 從指定 collection 的每份文件讀取 r 字串，提取最後 10 hex → Tag72
// ===========================================================================
static size_t load_collection(mongoc_client_t *client,
                              const char *db,
                              const char *collection,
                              std::vector<Tag72> &out) {
    printf("source: %s/%s.r\n", db, collection);
    printf("field: r (fallback r: r)\n");

    mongoc_collection_t *col = mongoc_client_get_collection(client, db, collection);

    bson_t *query = bson_new();
    bson_t *opts  = bson_new();
    BSON_APPEND_INT32(opts, "batchSize", 200000);

    // 只投影 r 欄位，減少網路傳輸量
    bson_t prj;
    BSON_APPEND_DOCUMENT_BEGIN(opts, "projection", &prj);
    BSON_APPEND_INT32(&prj, "r", 1);
    bson_append_document_end(opts, &prj);

    mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(col, query, opts, nullptr);
    const bson_t *doc = nullptr;

    size_t loaded = 0;
    size_t short_count = 0, nonhex_count = 0, no_field_count = 0, bad_type_count = 0;

    auto t0     = std::chrono::steady_clock::now();
    auto t_last = t0;

    while (mongoc_cursor_next(cursor, &doc)) {
        bson_iter_t it;
        const char *id_str = nullptr;
        uint32_t    id_len = 0;

        if (!bson_iter_init_find(&it, doc, "r")) {
            no_field_count++;
            continue;
        }

        char oid_hex[25] = {};
        if (BSON_ITER_HOLDS_UTF8(&it)) {
            id_str = bson_iter_utf8(&it, &id_len);
        } else if (BSON_ITER_HOLDS_OID(&it)) {
            bson_oid_to_string(bson_iter_oid(&it), oid_hex);
            id_str = oid_hex;
            id_len = 24;
        } else {
            bad_type_count++;
            continue;
        }

        if (!id_str || id_len < 10) { short_count++; continue; }

        Tag72 tag;
        if (!parse_last10hex(&tag, id_str, id_len)) { nonhex_count++; continue; }

        out.push_back(tag);
        loaded++;

        auto now = std::chrono::steady_clock::now();
        double since_print = std::chrono::duration<double>(now - t_last).count();
        if (since_print >= 2.0) {
            double total_elapsed = std::chrono::duration<double>(now - t0).count();
            double rate = (double)loaded / total_elapsed;
            printf("\r  [%-5s] %5zu M  (%.0f entries/s)   ",
                   collection, loaded / 1000000, rate);
            fflush(stdout);
            t_last = now;
        }
    }

    bson_error_t err;
    if (mongoc_cursor_error(cursor, &err)) {
        fprintf(stderr, "\n  cursor error in %s.%s: %s\n", db, collection, err.message);
        out.clear();
        loaded = 0;
    }

    mongoc_cursor_destroy(cursor);
    bson_destroy(opts);
    bson_destroy(query);
    mongoc_collection_destroy(col);

    auto t1      = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("\r  [%-5s] %zu entries loaded (%.1f s)\n", collection, loaded, elapsed);
    printf("         skipped: short=%zu nonhex=%zu no_field=%zu bad_type=%zu\n",
           short_count, nonhex_count, no_field_count, bad_type_count);
    return loaded;
}

// ===========================================================================
// build-part
// ===========================================================================
static int cmd_build_part(const char *collection, const char *part_file) {
    printf("=== build-part mode ===\n");

    const char *uri_str = getenv("MONGO_URI");
    if (!uri_str || uri_str[0] == '\0') uri_str = DEFAULT_MONGO_URI;

    mongoc_uri_t *uri = mongoc_uri_new(uri_str);
    if (!uri) { fprintf(stderr, "Invalid MongoDB URI\n"); return 1; }

    mongoc_client_t *client = mongoc_client_new_from_uri(uri);
    mongoc_uri_destroy(uri);
    if (!client) { fprintf(stderr, "Failed to create MongoDB client\n"); return 1; }

    mongoc_client_set_appname(client, "GenCache-build-part");

    std::vector<Tag72> entries;
    load_collection(client, DB_NAME, collection, entries);
    mongoc_client_destroy(client);

    if (entries.empty()) {
        fprintf(stderr, "save part failed\n");
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    printf("sorting %zu entries...\n", entries.size());
    std::sort(entries.begin(), entries.end(),
              [](const Tag72 &a, const Tag72 &b){ return tag72_less(a, b); });

    double sort_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("sorted in %.1f s\n", sort_time);

    size_t before = entries.size();
    entries.erase(
        std::unique(entries.begin(), entries.end(),
                    [](const Tag72 &a, const Tag72 &b){ return tag72_eq(a, b); }),
        entries.end());
    printf("dedup: %zu -> %zu\n", before, entries.size());

    if (!save_part_file(part_file, entries)) {
        fprintf(stderr, "save part failed\n");
        return 1;
    }
    printf("part saved: %s (%zu entries)\n", part_file, entries.size());
    return 0;
}

// ===========================================================================
// merge-parts  (k-way merge with priority queue - O(N log k))
// ===========================================================================
struct StreamPart {
    FILE    *fp    = nullptr;
    uint64_t total = 0;
    uint64_t pos   = 0;
    Tag72    buf   = {};
    bool     has_buf = false;

    bool open(const char *path) {
        fp = fopen(path, "rb");
        if (!fp) { fprintf(stderr, "Cannot open part file: %s\n", path); return false; }

        char magic[8];
        if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, PART_MAGIC, 8) != 0) {
            fprintf(stderr, "Invalid part header: %s\n", path);
            fclose(fp); fp = nullptr; return false;
        }
        if (fread(&total, sizeof(total), 1, fp) != 1) {
            fprintf(stderr, "Part format mismatch: %s\n", path);
            fclose(fp); fp = nullptr; return false;
        }
        pos = 0;
        return true;
    }

    bool peek(Tag72 &out) {
        if (has_buf) { out = buf; return true; }
        if (!fp || pos >= total) return false;
        if (fread(&buf, sizeof(Tag72), 1, fp) != 1) return false;
        pos++;
        has_buf = true;
        out = buf;
        return true;
    }

    void consume() { has_buf = false; }

    void close() { if (fp) { fclose(fp); fp = nullptr; } }
};

struct SPQItem {
    Tag72 tag;
    int   part_idx;
};

struct SPQCmp {
    // min-heap: 小的優先
    bool operator()(const SPQItem &a, const SPQItem &b) const {
        return tag72_less(b.tag, a.tag);
    }
};

static int cmd_merge_parts(const char *output_file,
                           int n_parts, const char **part_files) {
    printf("=== merge-parts mode ===\n");

    std::vector<StreamPart> parts((size_t)n_parts);
    uint64_t total_input = 0;
    for (int i = 0; i < n_parts; i++) {
        if (!parts[(size_t)i].open(part_files[i])) return 1;
        printf("opened part[%d]: %s (%llu entries)\n",
               i, part_files[i],
               (unsigned long long)parts[(size_t)i].total);
        total_input += parts[(size_t)i].total;
    }

    // 開啟輸出檔，先寫佔位 header（count 後面再補）
    FILE *out_fp = fopen(output_file, "wb");
    if (!out_fp) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        return 1;
    }
    const uint64_t placeholder_count = 0;
    if (fwrite(CACHE_MAGIC, 1, 8, out_fp) != 8 ||
        fwrite(&placeholder_count, sizeof(placeholder_count), 1, out_fp) != 1) {
        fprintf(stderr, "Cannot write output header\n");
        fclose(out_fp);
        return 1;
    }

    // 初始化 priority queue
    using PQ = std::priority_queue<SPQItem, std::vector<SPQItem>, SPQCmp>;
    PQ pq;
    for (int i = 0; i < n_parts; i++) {
        Tag72 t;
        if (parts[(size_t)i].peek(t)) {
            parts[(size_t)i].consume();
            SPQItem it; it.tag = t; it.part_idx = i;
            pq.push(it);
        }
    }

    uint64_t out_count = 0;
    Tag72    last      = {};
    bool     has_last  = false;

    auto t0     = std::chrono::steady_clock::now();
    auto t_last = t0;

    static const size_t FLUSH_BUF = 65536;
    std::vector<Tag72> wbuf;
    wbuf.reserve(FLUSH_BUF);

    auto flush_buf = [&]() -> bool {
        if (wbuf.empty()) return true;
        bool ok = fwrite(wbuf.data(), sizeof(Tag72), wbuf.size(), out_fp) == wbuf.size();
        wbuf.clear();
        return ok;
    };

    while (!pq.empty()) {
        SPQItem item = pq.top(); pq.pop();

        // 從同一個 part 補充下一筆
        Tag72 next;
        if (parts[(size_t)item.part_idx].peek(next)) {
            parts[(size_t)item.part_idx].consume();
            SPQItem nxt; nxt.tag = next; nxt.part_idx = item.part_idx;
            pq.push(nxt);
        }

        // 去重
        if (has_last && tag72_eq(item.tag, last)) continue;
        last     = item.tag;
        has_last = true;

        wbuf.push_back(item.tag);
        out_count++;

        if (wbuf.size() >= FLUSH_BUF) {
            if (!flush_buf()) {
                fprintf(stderr, "Write error\n");
                fclose(out_fp);
                return 1;
            }
        }

        auto now = std::chrono::steady_clock::now();
        double since_print = std::chrono::duration<double>(now - t_last).count();
        if (since_print >= 2.0) {
            double total_elapsed = std::chrono::duration<double>(now - t0).count();
            double speed_m = (double)out_count / total_elapsed / 1e6;
            printf("\r  merged %llu M / ~%llu M  (%.0f M/s)  ",
                   (unsigned long long)(out_count  / 1000000),
                   (unsigned long long)(total_input / 1000000),
                   speed_m);
            fflush(stdout);
            t_last = now;
        }
    }

    if (!flush_buf()) {
        fprintf(stderr, "Write error on final flush\n");
        fclose(out_fp);
        return 1;
    }

    // 回填正確的 count
    if (fseek(out_fp, 8, SEEK_SET) != 0 ||
        fwrite(&out_count, sizeof(out_count), 1, out_fp) != 1) {
        fprintf(stderr, "Cannot update output count\n");
        fclose(out_fp);
        return 1;
    }
    fclose(out_fp);

    for (int i = 0; i < n_parts; i++) parts[(size_t)i].close();

    printf("\nmerge done: input=%llu entries, output=%llu unique entries\n",
           (unsigned long long)total_input,
           (unsigned long long)out_count);
    return 0;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char **argv) {
    if (argc < 3) { print_usage(); return 1; }

    mongoc_init();
    int ret = 1;

    if (strcmp(argv[1], "build-part") == 0 && argc == 4) {
        ret = cmd_build_part(argv[2], argv[3]);
    } else if (strcmp(argv[1], "merge-parts") == 0 && argc >= 4) {
        ret = cmd_merge_parts(argv[2], argc - 3, (const char **)(argv + 3));
    } else {
        print_usage();
    }

    mongoc_cleanup();
    return ret;
}
