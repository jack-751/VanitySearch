/*
 * GenCacheMongo.cpp
 *
 * Build cache bin from MongoDB string field suffix.
 * Output format:
 *   magic[8] = "BTCTA%03u"  (e.g. BTCTA048 / BTCTA056 / BTCTA064)
 *   count[8] = uint64_t
 *   payload  = sorted unique tags, little-endian fixed-width (bits/8 bytes each)
 *
 * Compatible with current CalcR_GPU when bits=48 (hex-len=12).
 *
 * Build:
 *   g++ -O3 -std=c++17 -o GenCacheMongo GenCacheMongo.cpp \
 *       $(pkg-config --cflags --libs libmongoc-1.0)
 *
 * Example:
 *   ./GenCacheMongo \
 *     --mongo 192.168.50.171:27017 \
 *     --db ECDSA_BTC \
 *     --collection BTC \
 *     --field r \
 *     --hex-len 12 \
 *     --out mongo_keys.cache.bin
 */

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

struct Args {
    std::string mongo;
    std::string db;
    std::string collection;
    std::string field = "r";
    int hex_len = 12;          // 12->48 bits, 14->56 bits, 16->64 bits
    int batch_size = 200000;
    std::string out;
};

static void usage(const char *prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --mongo <host:port|mongodb://...> --db <db> --collection <name> --field <name> --hex-len <12|14|16> --out <file> [--batch-size N]\n",
        prog);
}

static bool parse_int(const char *s, int *out) {
    if (!s || !*s || !out) return false;
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = (int)v;
    return true;
}

static bool parse_args(int argc, char **argv, Args *a) {
    if (!a) return false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--mongo") == 0 && i + 1 < argc) a->mongo = argv[++i];
        else if (std::strcmp(argv[i], "--db") == 0 && i + 1 < argc) a->db = argv[++i];
        else if (std::strcmp(argv[i], "--collection") == 0 && i + 1 < argc) a->collection = argv[++i];
        else if (std::strcmp(argv[i], "--field") == 0 && i + 1 < argc) a->field = argv[++i];
        else if (std::strcmp(argv[i], "--hex-len") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &a->hex_len)) return false;
        }
        else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &a->batch_size)) return false;
        }
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) a->out = argv[++i];
        else return false;
    }

    if (a->mongo.empty() || a->db.empty() || a->collection.empty() || a->out.empty()) return false;
    if (!(a->hex_len == 12 || a->hex_len == 14 || a->hex_len == 16)) return false;
    if (a->batch_size <= 0) return false;

    if (a->mongo.rfind("mongodb://", 0) != 0 && a->mongo.rfind("mongodb+srv://", 0) != 0) {
        a->mongo = "mongodb://" + a->mongo;
    }
    if (a->mongo.find('?') == std::string::npos) {
        a->mongo += "/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=30000&socketTimeoutMS=600000";
    }
    return true;
}

static bool parse_suffix_hex_u64(const char *s, uint32_t len, int hex_len, uint64_t *out) {
    if (!s || !out || (int)len < hex_len) return false;
    const char *p = s + (len - hex_len);

    uint64_t v = 0;
    for (int i = 0; i < hex_len; i++) {
        const unsigned char c = (unsigned char)p[i];
        uint64_t nibble = 0;
        if (c >= '0' && c <= '9') nibble = (uint64_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (uint64_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (uint64_t)(c - 'A' + 10);
        else return false;
        v = (v << 4) | nibble;
    }

    *out = v;
    return true;
}

static inline void u64_to_le_bytes(uint64_t v, uint8_t *dst, int nbytes) {
    for (int i = 0; i < nbytes; i++) dst[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}

static inline uint64_t le_bytes_to_u64(const uint8_t *src, int nbytes) {
    uint64_t v = 0;
    for (int i = 0; i < nbytes; i++) v |= ((uint64_t)src[i]) << (8 * i);
    return v;
}

static bool write_exact(FILE *f, const void *buf, size_t size, size_t count) {
    return std::fwrite(buf, size, count, f) == count;
}

static std::string tmp_bucket_path(const std::string &out, int idx) {
    char b[1024];
    std::snprintf(b, sizeof(b), "%s.bucket.%03d.tmp", out.c_str(), idx);
    return std::string(b);
}

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 1;
    }

    const int bits = args.hex_len * 4;
    const int out_bytes = bits / 8;
    const int tail_bits = bits - 8;
    const int tail_bytes = tail_bits / 8;

    std::printf("=== GenCacheMongo ===\n");
    std::printf("mongo      : %s\n", args.mongo.c_str());
    std::printf("db/col/field: %s/%s/%s\n", args.db.c_str(), args.collection.c_str(), args.field.c_str());
    std::printf("hex-len    : %d (bits=%d)\n", args.hex_len, bits);
    std::printf("output     : %s\n", args.out.c_str());
    std::fflush(stdout);

    mongoc_init();

    mongoc_uri_t *uri = mongoc_uri_new(args.mongo.c_str());
    if (!uri) {
        std::fprintf(stderr, "Invalid Mongo URI\n");
        mongoc_cleanup();
        return 1;
    }

    mongoc_client_t *client = mongoc_client_new_from_uri(uri);
    mongoc_uri_destroy(uri);
    if (!client) {
        std::fprintf(stderr, "Failed to create Mongo client\n");
        mongoc_cleanup();
        return 1;
    }
    mongoc_client_set_appname(client, "GenCacheMongo");

    mongoc_collection_t *col = mongoc_client_get_collection(client, args.db.c_str(), args.collection.c_str());
    bson_t *query = bson_new();
    bson_t *opts = bson_new();
    BSON_APPEND_INT32(opts, "batchSize", args.batch_size);

    bson_t prj;
    BSON_APPEND_DOCUMENT_BEGIN(opts, "projection", &prj);
    BSON_APPEND_INT32(&prj, args.field.c_str(), 1);
    bson_append_document_end(opts, &prj);

    mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(col, query, opts, nullptr);

    std::vector<std::string> bucket_paths(256);
    std::vector<FILE*> bucket_fp(256, nullptr);
    for (int i = 0; i < 256; i++) {
        bucket_paths[i] = tmp_bucket_path(args.out, i);
        bucket_fp[i] = std::fopen(bucket_paths[i].c_str(), "wb");
        if (!bucket_fp[i]) {
            std::fprintf(stderr, "Cannot open bucket file: %s\n", bucket_paths[i].c_str());
            for (int j = 0; j <= i; j++) if (bucket_fp[j]) std::fclose(bucket_fp[j]);
            mongoc_cursor_destroy(cursor);
            bson_destroy(opts);
            bson_destroy(query);
            mongoc_collection_destroy(col);
            mongoc_client_destroy(client);
            mongoc_cleanup();
            return 1;
        }
    }

    const bson_t *doc = nullptr;
    uint64_t total_docs = 0;
    uint64_t ok_tags = 0;
    uint64_t skip_no_field = 0, skip_bad_type = 0, skip_short = 0, skip_nonhex = 0;

    auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

    std::vector<uint8_t> tb((size_t)tail_bytes);

    while (mongoc_cursor_next(cursor, &doc)) {
        total_docs++;

        bson_iter_t it;
        if (!bson_iter_init_find(&it, doc, args.field.c_str())) {
            skip_no_field++;
            continue;
        }
        if (!BSON_ITER_HOLDS_UTF8(&it)) {
            skip_bad_type++;
            continue;
        }

        uint32_t slen = 0;
        const char *s = bson_iter_utf8(&it, &slen);
        if (!s || (int)slen < args.hex_len) {
            skip_short++;
            continue;
        }

        uint64_t v = 0;
        if (!parse_suffix_hex_u64(s, slen, args.hex_len, &v)) {
            skip_nonhex++;
            continue;
        }

        const uint8_t bucket = (uint8_t)(v >> tail_bits);
        const uint64_t tail = (tail_bits == 64) ? v : (v & ((1ULL << tail_bits) - 1ULL));
        u64_to_le_bytes(tail, tb.data(), tail_bytes);
        if (!write_exact(bucket_fp[bucket], tb.data(), (size_t)tail_bytes, 1)) {
            std::fprintf(stderr, "write bucket failed at bucket %u\n", (unsigned)bucket);
            for (int i = 0; i < 256; i++) if (bucket_fp[i]) std::fclose(bucket_fp[i]);
            mongoc_cursor_destroy(cursor);
            bson_destroy(opts);
            bson_destroy(query);
            mongoc_collection_destroy(col);
            mongoc_client_destroy(client);
            mongoc_cleanup();
            return 1;
        }

        ok_tags++;

        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - t_last).count();
        if (since >= 2.0) {
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double rate = elapsed > 0 ? ((double)total_docs / elapsed) : 0;
            std::printf("\rscan docs=%llu tags=%llu rate=%.0f docs/s", (unsigned long long)total_docs, (unsigned long long)ok_tags, rate);
            std::fflush(stdout);
            t_last = now;
        }
    }
    std::printf("\nscan complete: docs=%llu tags=%llu\n", (unsigned long long)total_docs, (unsigned long long)ok_tags);
    std::printf("skipped: no_field=%llu bad_type=%llu short=%llu nonhex=%llu\n",
        (unsigned long long)skip_no_field,
        (unsigned long long)skip_bad_type,
        (unsigned long long)skip_short,
        (unsigned long long)skip_nonhex);

    bson_error_t cerr;
    if (mongoc_cursor_error(cursor, &cerr)) {
        std::fprintf(stderr, "Mongo cursor error: %s\n", cerr.message);
        for (int i = 0; i < 256; i++) if (bucket_fp[i]) std::fclose(bucket_fp[i]);
        mongoc_cursor_destroy(cursor);
        bson_destroy(opts);
        bson_destroy(query);
        mongoc_collection_destroy(col);
        mongoc_client_destroy(client);
        mongoc_cleanup();
        return 1;
    }

    for (int i = 0; i < 256; i++) {
        std::fclose(bucket_fp[i]);
        bucket_fp[i] = nullptr;
    }

    mongoc_cursor_destroy(cursor);
    bson_destroy(opts);
    bson_destroy(query);
    mongoc_collection_destroy(col);
    mongoc_client_destroy(client);
    mongoc_cleanup();

    FILE *out = std::fopen(args.out.c_str(), "wb");
    if (!out) {
        std::fprintf(stderr, "Cannot open output file: %s\n", args.out.c_str());
        return 1;
    }

    char magic[9] = {0};
    std::snprintf(magic, sizeof(magic), "BTCTA%03d", bits);
    uint64_t final_count = 0;

    if (!write_exact(out, magic, 1, 8) || !write_exact(out, &final_count, sizeof(final_count), 1)) {
        std::fprintf(stderr, "Failed writing output header\n");
        std::fclose(out);
        return 1;
    }

    std::vector<uint8_t> read_buf;
    std::vector<uint64_t> vals;
    std::vector<uint8_t> out_tag((size_t)out_bytes);

    for (int b = 0; b < 256; b++) {
        const std::string &bp = bucket_paths[b];
        FILE *bf = std::fopen(bp.c_str(), "rb");
        if (!bf) continue;

        if (std::fseek(bf, 0, SEEK_END) != 0) {
            std::fprintf(stderr, "fseek failed: %s\n", bp.c_str());
            std::fclose(bf);
            std::fclose(out);
            return 1;
        }
        long long szll = (long long)std::ftell(bf);
        if (szll < 0) {
            std::fprintf(stderr, "ftell failed: %s\n", bp.c_str());
            std::fclose(bf);
            std::fclose(out);
            return 1;
        }
        std::rewind(bf);

        const uint64_t sz = (uint64_t)szll;
        if (sz == 0) {
            std::fclose(bf);
            std::remove(bp.c_str());
            continue;
        }
        if (sz % (uint64_t)tail_bytes != 0) {
            std::fprintf(stderr, "bucket size mismatch: %s\n", bp.c_str());
            std::fclose(bf);
            std::fclose(out);
            return 1;
        }

        const uint64_t n = sz / (uint64_t)tail_bytes;
        vals.clear();
        vals.reserve((size_t)n);

        const size_t CHUNK_ELEMS = 1u << 20;
        read_buf.resize(CHUNK_ELEMS * (size_t)tail_bytes);

        uint64_t done = 0;
        while (done < n) {
            size_t take = (size_t)((n - done) > CHUNK_ELEMS ? CHUNK_ELEMS : (n - done));
            size_t got = std::fread(read_buf.data(), (size_t)tail_bytes, take, bf);
            if (got != take) {
                std::fprintf(stderr, "bucket read failed: %s\n", bp.c_str());
                std::fclose(bf);
                std::fclose(out);
                return 1;
            }
            for (size_t i = 0; i < got; i++) {
                vals.push_back(le_bytes_to_u64(&read_buf[i * (size_t)tail_bytes], tail_bytes));
            }
            done += got;
        }
        std::fclose(bf);

        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        for (size_t i = 0; i < vals.size(); i++) {
            const uint64_t full = ((uint64_t)b << tail_bits) | vals[i];
            u64_to_le_bytes(full, out_tag.data(), out_bytes);
            if (!write_exact(out, out_tag.data(), (size_t)out_bytes, 1)) {
                std::fprintf(stderr, "write output failed at bucket=%d\n", b);
                std::fclose(out);
                return 1;
            }
            final_count++;
        }

        std::remove(bp.c_str());
        std::printf("bucket %3d done: in=%llu unique=%zu\n", b, (unsigned long long)n, vals.size());
        std::fflush(stdout);
    }

    if (std::fseek(out, 8, SEEK_SET) != 0 || !write_exact(out, &final_count, sizeof(final_count), 1)) {
        std::fprintf(stderr, "patch count failed\n");
        std::fclose(out);
        return 1;
    }
    std::fclose(out);

    std::printf("\nDone. output=%s magic=%s count=%llu entry_bytes=%d\n",
        args.out.c_str(), magic, (unsigned long long)final_count, out_bytes);
    return 0;
}
