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
    // 開啟輸出檔案
    // =====================
    FILE *out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Error: cannot open output file: %s\n", output_file);
        fclose(in);
        return 1;
    }

    // 寫入新 magic
    const char out_magic[8] = {'B', 'T', 'C', 'T', 'A', '0', '4', '0'};
    if (fwrite(out_magic, 1, 8, out) != 8) {
        fprintf(stderr, "Error: cannot write output magic\n");
        fclose(in);
        fclose(out);
        return 1;
    }

    // 寫入新 count（不寫 version）
    if (fwrite(&count, sizeof(count), 1, out) != 1) {
        fprintf(stderr, "Error: cannot write output count\n");
        fclose(in);
        fclose(out);
        return 1;
    }

    // =====================
    // 轉換資料
    // =====================
    const size_t BATCH_SIZE = 1000000;  // 一次轉換 1M 個 tag，降低記憶體使用
    std::vector<Tag96> in_batch(BATCH_SIZE);
    std::vector<Tag40> out_batch(BATCH_SIZE);

    uint64_t processed = 0;
    auto last_print_time = std::chrono::high_resolution_clock::now();

    while (processed < count) {
        size_t to_read = std::min((uint64_t)BATCH_SIZE, count - processed);
        size_t read_count = fread(in_batch.data(), sizeof(Tag96), to_read, in);
        
        if (read_count != to_read) {
            fprintf(stderr, "Error: read mismatch at offset %llu\n", (unsigned long long)processed);
            break;
        }

        // 轉換：從 96-bit (24 hex) 提取最後 10 hex
        // 96 bits = 12 hex (32-bit hi) + 16 hex (64-bit lo)
        // 最後 10 hex 全部都在 lo(64-bit) 之內：
        //   hi = lo 的 bits 32..39
        //   lo = lo 的 bits 0..31
        for (size_t i = 0; i < read_count; i++) {
            out_batch[i].lo = (uint32_t)(in_batch[i].lo & 0xFFFFFFFFULL);
            out_batch[i].hi = (uint8_t)((in_batch[i].lo >> 32) & 0xFFULL);
        }

        // 寫出轉換後的資料
        size_t written = fwrite(out_batch.data(), sizeof(Tag40), read_count, out);
        if (written != read_count) {
            fprintf(stderr, "Error: write mismatch at offset %llu\n", (unsigned long long)processed);
            break;
        }

        processed += read_count;

        // 進度輸出
        auto now = std::chrono::high_resolution_clock::now();
        double since_print = std::chrono::duration<double>(now - last_print_time).count();
        if (since_print >= 2.0) {
            double percent = (double)processed / count * 100.0;
            printf("\r  Progress: %.1f%% (%llu / %llu)", percent,
                   (unsigned long long)processed, (unsigned long long)count);
            fflush(stdout);
            last_print_time = now;
        }
    }

    printf("\n✓ Conversion complete: %llu tags processed\n", (unsigned long long)processed);

    fclose(in);
    fclose(out);

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
