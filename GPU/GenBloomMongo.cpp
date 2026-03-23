/*
 * GenBloomMongo.cpp
 *
 * Build blocked bloom filter from MongoDB field suffix.
 * Supports suffix hex length: 12..20.
 *
 * Build:
 *   g++ -O3 -std=c++17 -o GenBloomMongo GenBloomMongo.cpp \
 *     $(pkg-config --cflags --libs libmongoc-1.0)
 */

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

#include "BlockedBloom.h"

struct Args {
    std::string mongo;
    std::string db;
    std::string collection;
    std::string field = "r";
    int hex_len = 12;
    int batch_size = 200000;
    uint64_t expected_items = 2000000000ULL;
    double fp_rate = 1e-4;
    std::string out;
};

static void usage(const char *prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --mongo <host:port|mongodb://...> --db <db> --collection <name> --field <name> --hex-len <12..20> --out <file> [--batch-size N] [--expected-items N] [--fp-rate F]\n",
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

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s || !out) return false;
    char *end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

static bool parse_double(const char *s, double *out) {
    if (!s || !*s || !out) return false;
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    *out = v;
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
        else if (std::strcmp(argv[i], "--expected-items") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &a->expected_items)) return false;
        }
        else if (std::strcmp(argv[i], "--fp-rate") == 0 && i + 1 < argc) {
            if (!parse_double(argv[++i], &a->fp_rate)) return false;
        }
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) a->out = argv[++i];
        else return false;
    }

    if (a->mongo.empty() || a->db.empty() || a->collection.empty() || a->out.empty()) return false;
    if (!(a->hex_len >= 12 && a->hex_len <= 20)) return false;
    if (a->batch_size <= 0 || a->expected_items == 0) return false;
    if (!(a->fp_rate > 0.0 && a->fp_rate < 1.0)) return false;

    if (a->mongo.rfind("mongodb://", 0) != 0 && a->mongo.rfind("mongodb+srv://", 0) != 0) {
        a->mongo = "mongodb://" + a->mongo;
    }
    if (a->mongo.find('?') == std::string::npos) {
        a->mongo += "/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=30000&socketTimeoutMS=600000";
    }
    return true;
}

static bool parse_suffix_hex_le(const char *s, uint32_t len, int hex_len, uint8_t *out, size_t out_sz) {
    if (!s || !out || out_sz < (size_t)((hex_len + 1) / 2) || (int)len < hex_len) return false;
    std::memset(out, 0, out_sz);

    const char *p = s + (len - hex_len);
    int nibble_idx = 0;
    for (int i = hex_len - 1; i >= 0; i--) {
        const unsigned char c = (unsigned char)p[i];
        uint8_t v = 0;
        if (c >= '0' && c <= '9') v = (uint8_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v = (uint8_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = (uint8_t)(c - 'A' + 10);
        else return false;

        const int byte_idx = nibble_idx >> 1;
        if ((nibble_idx & 1) == 0) out[byte_idx] = v;
        else out[byte_idx] |= (uint8_t)(v << 4);
        nibble_idx++;
    }
    return true;
}

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 1;
    }

    std::printf("=== GenBloomMongo ===\n");
    std::printf("mongo       : %s\n", args.mongo.c_str());
    std::printf("db/col/field: %s/%s/%s\n", args.db.c_str(), args.collection.c_str(), args.field.c_str());
    std::printf("hex-len     : %d\n", args.hex_len);
    std::printf("expected    : %llu\n", (unsigned long long)args.expected_items);
    std::printf("fp-rate     : %.8f\n", args.fp_rate);
    std::printf("output      : %s\n", args.out.c_str());

    BloomConfig cfg;
    cfg.expected_items = args.expected_items;
    cfg.fp_rate = args.fp_rate;
    cfg.block_bits = 512;

    BlockedBloomFilter bloom;
    if (!bloom.init((uint32_t)args.hex_len, cfg)) {
        std::fprintf(stderr, "Failed to init bloom\n");
        return 1;
    }

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
    mongoc_client_set_appname(client, "GenBloomMongo");

    mongoc_collection_t *col = mongoc_client_get_collection(client, args.db.c_str(), args.collection.c_str());
    bson_t *query = bson_new();
    bson_t *opts = bson_new();
    BSON_APPEND_INT32(opts, "batchSize", args.batch_size);

    bson_t prj;
    BSON_APPEND_DOCUMENT_BEGIN(opts, "projection", &prj);
    BSON_APPEND_INT32(&prj, args.field.c_str(), 1);
    bson_append_document_end(opts, &prj);

    mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(col, query, opts, nullptr);

    const bson_t *doc = nullptr;
    uint64_t total_docs = 0;
    uint64_t inserted = 0;
    uint64_t skip_no_field = 0, skip_bad_type = 0, skip_short = 0, skip_nonhex = 0;

    std::vector<uint8_t> tag_bytes((size_t)((args.hex_len + 1) / 2));

    auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

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

        if (!parse_suffix_hex_le(s, slen, args.hex_len, tag_bytes.data(), tag_bytes.size())) {
            skip_nonhex++;
            continue;
        }

        bloom.add(tag_bytes.data(), tag_bytes.size());
        inserted++;

        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - t_last).count();
        if (since >= 2.0) {
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double rate = elapsed > 0 ? ((double)total_docs / elapsed) : 0;
            std::printf("\rscan docs=%llu inserted=%llu rate=%.0f docs/s", (unsigned long long)total_docs, (unsigned long long)inserted, rate);
            std::fflush(stdout);
            t_last = now;
        }
    }
    std::printf("\nscan complete: docs=%llu inserted=%llu\n", (unsigned long long)total_docs, (unsigned long long)inserted);
    std::printf("skipped: no_field=%llu bad_type=%llu short=%llu nonhex=%llu\n",
        (unsigned long long)skip_no_field,
        (unsigned long long)skip_bad_type,
        (unsigned long long)skip_short,
        (unsigned long long)skip_nonhex);

    bson_error_t cerr;
    if (mongoc_cursor_error(cursor, &cerr)) {
        std::fprintf(stderr, "Mongo cursor error: %s\n", cerr.message);
        mongoc_cursor_destroy(cursor);
        bson_destroy(opts);
        bson_destroy(query);
        mongoc_collection_destroy(col);
        mongoc_client_destroy(client);
        mongoc_cleanup();
        return 1;
    }

    mongoc_cursor_destroy(cursor);
    bson_destroy(opts);
    bson_destroy(query);
    mongoc_collection_destroy(col);
    mongoc_client_destroy(client);
    mongoc_cleanup();

    if (!bloom.save(args.out.c_str())) {
        std::fprintf(stderr, "Failed to save bloom file: %s\n", args.out.c_str());
        return 1;
    }

    std::printf("Done. bloom=%s inserted=%llu k=%u blocks=%llu\n",
        args.out.c_str(),
        (unsigned long long)bloom.insertedItems(),
        bloom.kHashes(),
        (unsigned long long)bloom.numBlocks());
    return 0;
}
