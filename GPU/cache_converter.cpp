/*
 * cache_converter.cpp
 *
 * 將 96-bit 快取 (BTCTA096 format) 轉換為 40-bit 快取 (BTCTA040 format)
 * 提取最後 10 hex 字元（而不是 24 個）
 *
 * 編譯：
 *   g++ -O2 -std=c++17 -o cache_converter cache_converter.cpp
 *
 * 用法：
 *   ./cache_converter <input_96bit.bin> <output_40bit.bin>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <queue>
#include <limits>
#include <unistd.h>

// 96-bit tag (old format): uint32_t hi (32 bits) + uint64_t lo (64 bits)
struct Tag96 {
    uint32_t hi;
    uint64_t lo;
} __attribute__((packed));

// 40-bit tag (new format): uint8_t hi (8 bits) + uint32_t lo (32 bits)
struct Tag40 {
    uint32_t lo;
    uint8_t  hi;
} __attribute__((packed));

static bool read_exact(FILE *f, void *buf, size_t elem_size, size_t count) {
    return fread(buf, elem_size, count, f) == count;
}

static bool write_exact(FILE *f, const void *buf, size_t elem_size, size_t count) {
    return fwrite(buf, elem_size, count, f) == count;
}

static std::string make_temp_prefix(const char *output_file) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s.tmp.%d.%llu",
                  output_file,
                  (int)getpid(),
                  (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return std::string(buf);
}

static std::string bucket_path(const std::string &prefix, int hi) {
    char buf[640];
    std::snprintf(buf, sizeof(buf), "%s.bucket.%03d.bin", prefix.c_str(), hi);
    return std::string(buf);
}

static std::string run_path(const std::string &prefix, int hi, size_t run_idx) {
    char buf[640];
    std::snprintf(buf, sizeof(buf), "%s.run.%03d.%06zu.bin", prefix.c_str(), hi, run_idx);
    return std::string(buf);
}

struct MergeNode {
    uint32_t value;
    size_t run_idx;
    bool operator>(const MergeNode &o) const {
        if (value != o.value) return value > o.value;
        return run_idx > o.run_idx;
    }
};

struct RunReader {
    FILE *f = nullptr;
};

static bool process_single_bucket(
    const std::string &bucket_file,
    const std::string &tmp_prefix,
    int hi,
    FILE *out,
    uint64_t *out_count)
{
    FILE *in = fopen(bucket_file.c_str(), "rb");
    if (!in) {
        return true;
    }

    if (fseek(in, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error: fseek failed for bucket %d\n", hi);
        fclose(in);
        return false;
    }
    long long bytes = (long long)ftell(in);
    if (bytes < 0) {
        fprintf(stderr, "Error: ftell failed for bucket %d\n", hi);
        fclose(in);
        return false;
    }
    rewind(in);

    if (bytes == 0) {
        fclose(in);
        remove(bucket_file.c_str());
        return true;
    }

    const size_t CHUNK_ELEMS = 8ULL * 1024ULL * 1024ULL;
    std::vector<uint32_t> chunk;
    chunk.resize(CHUNK_ELEMS);

    std::vector<std::string> runs;
    size_t run_idx = 0;
    while (true) {
        size_t got = fread(chunk.data(), sizeof(uint32_t), CHUNK_ELEMS, in);
        if (got == 0) break;

        std::sort(chunk.begin(), chunk.begin() + got);
        std::string rp = run_path(tmp_prefix, hi, run_idx++);
        FILE *rf = fopen(rp.c_str(), "wb");
        if (!rf) {
            fprintf(stderr, "Error: cannot create run file: %s\n", rp.c_str());
            fclose(in);
            return false;
        }
        if (!write_exact(rf, chunk.data(), sizeof(uint32_t), got)) {
            fprintf(stderr, "Error: write run failed: %s\n", rp.c_str());
            fclose(rf);
            fclose(in);
            return false;
        }
        fclose(rf);
        runs.push_back(std::move(rp));
    }
    fclose(in);
    remove(bucket_file.c_str());

    std::vector<RunReader> readers(runs.size());
    std::priority_queue<MergeNode, std::vector<MergeNode>, std::greater<MergeNode>> pq;

    for (size_t i = 0; i < runs.size(); i++) {
        readers[i].f = fopen(runs[i].c_str(), "rb");
        if (!readers[i].f) {
            fprintf(stderr, "Error: cannot open run file for merge: %s\n", runs[i].c_str());
            return false;
        }
        uint32_t v;
        if (read_exact(readers[i].f, &v, sizeof(v), 1)) {
            pq.push(MergeNode{v, i});
        }
    }

    bool has_last = false;
    uint32_t last = 0;
    Tag40 out_tag{};
    out_tag.hi = (uint8_t)hi;

    while (!pq.empty()) {
        MergeNode cur = pq.top();
        pq.pop();

        if (!has_last || cur.value != last) {
            out_tag.lo = cur.value;
            if (!write_exact(out, &out_tag, sizeof(out_tag), 1)) {
                fprintf(stderr, "Error: write output tag failed (hi=%d)\n", hi);
                for (size_t j = 0; j < readers.size(); j++) {
                    if (readers[j].f) fclose(readers[j].f);
                }
                return false;
            }
            (*out_count)++;
            last = cur.value;
            has_last = true;
        }

        uint32_t next_v;
        if (read_exact(readers[cur.run_idx].f, &next_v, sizeof(next_v), 1)) {
            pq.push(MergeNode{next_v, cur.run_idx});
        }
    }

    for (size_t i = 0; i < readers.size(); i++) {
        if (readers[i].f) fclose(readers[i].f);
        remove(runs[i].c_str());
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_96bit.bin> <output_40bit.bin>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    printf("Converting cache from 96-bit to 40-bit...\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);

    // =====================
    // 讀取輸入檔案頭部
    // =====================
    FILE *in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Error: cannot open input file: %s\n", input_file);
        return 1;
    }

    // 讀取 magic
    char magic[8];
    if (fread(magic, 1, 8, in) != 8) {
        fprintf(stderr, "Error: cannot read magic\n");
        fclose(in);
        return 1;
    }

    printf("Input magic: %.8s\n", magic);
    if (memcmp(magic, "BTCTA096", 8) != 0) {
        fprintf(stderr, "Error: input magic is not BTCTA096\n");
        fclose(in);
        return 1;
    }

    // 讀取 version（8 bytes）
    uint64_t version;
    if (fread(&version, sizeof(version), 1, in) != 1) {
        fprintf(stderr, "Error: cannot read version\n");
        fclose(in);
        return 1;
    }
    printf("Input version: %llu\n", (unsigned long long)version);

    // 讀取 count（8 bytes）
    uint64_t count;
    if (fread(&count, sizeof(count), 1, in) != 1) {
        fprintf(stderr, "Error: cannot read count\n");
        fclose(in);
        return 1;
    }
    printf("Input tag count: %llu\n", (unsigned long long)count);

    // =====================
    // 第 1 階段：分桶
    // =====================
    std::string tmp_prefix = make_temp_prefix(output_file);
    std::vector<std::string> bucket_files(256);
    std::vector<FILE*> bucket_fp(256, nullptr);

    for (int i = 0; i < 256; i++) {
        bucket_files[i] = bucket_path(tmp_prefix, i);
        bucket_fp[i] = fopen(bucket_files[i].c_str(), "wb");
        if (!bucket_fp[i]) {
            fprintf(stderr, "Error: cannot create bucket file: %s\n", bucket_files[i].c_str());
            fclose(in);
            return 1;
        }
    }

    const size_t BATCH_SIZE = 1000000;
    std::vector<Tag96> in_batch(BATCH_SIZE);
    uint64_t processed = 0;
    auto last_print_time = std::chrono::high_resolution_clock::now();

    while (processed < count) {
        size_t to_read = std::min((uint64_t)BATCH_SIZE, count - processed);
        size_t read_count = fread(in_batch.data(), sizeof(Tag96), to_read, in);
        if (read_count != to_read) {
            fprintf(stderr, "Error: read mismatch at offset %llu\n", (unsigned long long)processed);
            break;
        }

        for (size_t i = 0; i < read_count; i++) {
            uint8_t hi = (uint8_t)((in_batch[i].lo >> 32) & 0xFFULL);
            uint32_t lo = (uint32_t)(in_batch[i].lo & 0xFFFFFFFFULL);
            if (!write_exact(bucket_fp[hi], &lo, sizeof(lo), 1)) {
                fprintf(stderr, "Error: write bucket failed (hi=%u)\n", (unsigned int)hi);
                fclose(in);
                return 1;
            }
        }

        processed += read_count;
        auto now = std::chrono::high_resolution_clock::now();
        double since_print = std::chrono::duration<double>(now - last_print_time).count();
        if (since_print >= 2.0) {
            double percent = (double)processed / count * 100.0;
            printf("\r  Bucketizing: %.1f%% (%llu / %llu)", percent,
                   (unsigned long long)processed, (unsigned long long)count);
            fflush(stdout);
            last_print_time = now;
        }
    }
    printf("\nBucketizing done: %llu tags processed\n", (unsigned long long)processed);

    fclose(in);
    for (int i = 0; i < 256; i++) {
        if (bucket_fp[i]) fclose(bucket_fp[i]);
    }

    if (processed != count) {
        fprintf(stderr, "Error: conversion aborted during bucketizing\n");
        return 1;
    }

    // =====================
    // 第 2 階段：每桶外部排序 + 去重，並按 hi 順序輸出
    // =====================
    FILE *out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Error: cannot open output file: %s\n", output_file);
        return 1;
    }

    const char out_magic[8] = {'B', 'T', 'C', 'T', 'A', '0', '4', '0'};
    uint64_t final_count = 0;
    if (!write_exact(out, out_magic, 1, 8) || !write_exact(out, &final_count, sizeof(final_count), 1)) {
        fprintf(stderr, "Error: cannot write output header\n");
        fclose(out);
        return 1;
    }

    for (int hi = 0; hi < 256; hi++) {
        printf("Sorting+dedup bucket %d/255...\n", hi);
        fflush(stdout);
        if (!process_single_bucket(bucket_files[hi], tmp_prefix, hi, out, &final_count)) {
            fclose(out);
            return 1;
        }
    }

    if (fseek(out, 8, SEEK_SET) != 0 || !write_exact(out, &final_count, sizeof(final_count), 1)) {
        fprintf(stderr, "Error: cannot patch final output count\n");
        fclose(out);
        return 1;
    }

    fclose(out);

    printf("\n✓ Conversion complete: %llu input tags, %llu unique output tags\n",
           (unsigned long long)processed,
           (unsigned long long)final_count);

    // 驗證輸出檔案
    printf("\nVerifying output file...\n");
    FILE *verify = fopen(output_file, "rb");
    if (!verify) {
        fprintf(stderr, "Error: cannot open output file for verification\n");
        return 1;
    }

    char verify_magic[8];
    uint64_t verify_count;
    if (fread(verify_magic, 1, 8, verify) != 8 ||
        fread(&verify_count, sizeof(verify_count), 1, verify) != 1) {
        fprintf(stderr, "Error: cannot read output header for verification\n");
        fclose(verify);
        return 1;
    }
    fclose(verify);

    printf("Output magic:  %.8s\n", verify_magic);
    printf("Output count:  %llu\n", (unsigned long long)verify_count);
        printf("Output format: 40-bit (10 hex chars per tag)\n");
    printf("Output file size: ~%llu bytes\n",
           (unsigned long long)(16 + verify_count * sizeof(Tag40)));

    return 0;
}
