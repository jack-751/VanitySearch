/*
 * cache_verify.cpp
 *
 * Verify BTCTA040 cache file integrity.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o cache_verify cache_verify.cpp
 *
 * Usage:
 *   ./cache_verify <cache_40bit.bin> [--full] [--sample N]
 *
 * Modes:
 *   - default (quick): header/size checks + sampled order checks
 *   - --full: full linear scan, verifies global non-decreasing order and counts adjacent duplicates
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <random>
#include <chrono>
#include <vector>

struct Tag40 {
    uint32_t lo;
    uint8_t hi;
} __attribute__((packed));

static int compare_tag(const Tag40 &a, const Tag40 &b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}

static bool read_tag_at(FILE *f, uint64_t index, Tag40 *out) {
    if (!f || !out) return false;
    const uint64_t off = 16ULL + index * (uint64_t)sizeof(Tag40);
    if (fseek(f, (long)off, SEEK_SET) != 0) return false;
    return fread(out, sizeof(Tag40), 1, f) == 1;
}

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s || !out) return false;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cache_40bit.bin> [--full] [--sample N]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    bool full = false;
    uint64_t sample_n = 1000000ULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (strcmp(argv[i], "--sample") == 0) {
            if (i + 1 >= argc || !parse_u64(argv[i + 1], &sample_n) || sample_n == 0) {
                fprintf(stderr, "Invalid --sample value\n");
                return 1;
            }
            i++;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file: %s\n", path);
        return 1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error: fseek failed\n");
        fclose(f);
        return 1;
    }
    long long ll_size = (long long)ftell(f);
    if (ll_size < 0) {
        fprintf(stderr, "Error: ftell failed\n");
        fclose(f);
        return 1;
    }
    uint64_t file_size = (uint64_t)ll_size;
    rewind(f);

    char magic[8] = {0};
    uint64_t count = 0;
    if (fread(magic, 1, 8, f) != 8 || fread(&count, sizeof(count), 1, f) != 1) {
        fprintf(stderr, "Error: cannot read header\n");
        fclose(f);
        return 1;
    }

    const char expected_magic[8] = {'B','T','C','T','A','0','4','0'};
    bool magic_ok = memcmp(magic, expected_magic, 8) == 0;
    uint64_t expected_size = 16ULL + count * (uint64_t)sizeof(Tag40);
    bool size_ok = (file_size == expected_size);

    printf("File: %s\n", path);
    printf("Magic: %.8s (%s)\n", magic, magic_ok ? "OK" : "BAD");
    printf("Count: %llu\n", (unsigned long long)count);
    printf("Size:  %llu bytes\n", (unsigned long long)file_size);
    printf("Expect:%llu bytes (%s)\n", (unsigned long long)expected_size, size_ok ? "OK" : "MISMATCH");

    if (!magic_ok || !size_ok) {
        fclose(f);
        return 2;
    }

    if (count <= 1) {
        printf("Trivial file: sorted by definition.\n");
        fclose(f);
        return 0;
    }

    bool sorted_ok = true;
    bool duplicate_seen = false;

    if (full) {
        printf("Mode: full scan\n");
        Tag40 prev, cur;
        if (!read_tag_at(f, 0, &prev)) {
            fprintf(stderr, "Error: cannot read first tag\n");
            fclose(f);
            return 1;
        }

        uint64_t dup_adj = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto last_print = start;

        for (uint64_t i = 1; i < count; i++) {
            if (!read_tag_at(f, i, &cur)) {
                fprintf(stderr, "Error: read failed at index %llu\n", (unsigned long long)i);
                fclose(f);
                return 1;
            }
            int cmp = compare_tag(prev, cur);
            if (cmp > 0) {
                sorted_ok = false;
                printf("Order break at index %llu: prev=(hi=%u, lo=%u), cur=(hi=%u, lo=%u)\n",
                       (unsigned long long)i,
                       (unsigned int)prev.hi, (unsigned int)prev.lo,
                       (unsigned int)cur.hi, (unsigned int)cur.lo);
                break;
            }
            if (cmp == 0) {
                dup_adj++;
                duplicate_seen = true;
            }
            prev = cur;

            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - last_print).count();
            if (dt >= 2.0) {
                double p = (double)i * 100.0 / (double)(count - 1);
                printf("\r  Progress: %.2f%% (%llu / %llu)", p,
                       (unsigned long long)i,
                       (unsigned long long)(count - 1));
                fflush(stdout);
                last_print = now;
            }
        }
        printf("\nAdjacent duplicates: %llu\n", (unsigned long long)dup_adj);
    } else {
        printf("Mode: quick sampled checks (sample=%llu)\n", (unsigned long long)sample_n);

        uint64_t head_n = sample_n;
        if (head_n > count - 1) head_n = count - 1;

        Tag40 prev, cur;
        if (!read_tag_at(f, 0, &prev)) {
            fprintf(stderr, "Error: cannot read first tag\n");
            fclose(f);
            return 1;
        }

        for (uint64_t i = 1; i <= head_n; i++) {
            if (!read_tag_at(f, i, &cur)) {
                fprintf(stderr, "Error: read failed at index %llu\n", (unsigned long long)i);
                fclose(f);
                return 1;
            }
            int cmp = compare_tag(prev, cur);
            if (cmp > 0) {
                sorted_ok = false;
                printf("Order break in head sample at index %llu: prev=(hi=%u, lo=%u), cur=(hi=%u, lo=%u)\n",
                       (unsigned long long)i,
                       (unsigned int)prev.hi, (unsigned int)prev.lo,
                       (unsigned int)cur.hi, (unsigned int)cur.lo);
                break;
            }
            if (cmp == 0) duplicate_seen = true;
            prev = cur;
        }

        if (sorted_ok) {
            uint64_t tail_start = 0;
            if (count > sample_n + 1) {
                tail_start = count - (sample_n + 1);
            }

            if (!read_tag_at(f, tail_start, &prev)) {
                fprintf(stderr, "Error: cannot read tail start\n");
                fclose(f);
                return 1;
            }

            for (uint64_t i = tail_start + 1; i < count; i++) {
                if (!read_tag_at(f, i, &cur)) {
                    fprintf(stderr, "Error: read failed at index %llu\n", (unsigned long long)i);
                    fclose(f);
                    return 1;
                }
                int cmp = compare_tag(prev, cur);
                if (cmp > 0) {
                    sorted_ok = false;
                    printf("Order break in tail sample at index %llu: prev=(hi=%u, lo=%u), cur=(hi=%u, lo=%u)\n",
                           (unsigned long long)i,
                           (unsigned int)prev.hi, (unsigned int)prev.lo,
                           (unsigned int)cur.hi, (unsigned int)cur.lo);
                    break;
                }
                if (cmp == 0) duplicate_seen = true;
                prev = cur;
            }
        }

        if (sorted_ok && count > 2) {
            std::mt19937_64 rng((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            uint64_t rand_checks = sample_n;
            if (rand_checks > count - 1) rand_checks = count - 1;

            for (uint64_t k = 0; k < rand_checks; k++) {
                uint64_t i = (rng() % (count - 1));
                Tag40 a, b;
                if (!read_tag_at(f, i, &a) || !read_tag_at(f, i + 1, &b)) {
                    fprintf(stderr, "Error: random check read failed at index %llu\n", (unsigned long long)i);
                    fclose(f);
                    return 1;
                }
                int cmp = compare_tag(a, b);
                if (cmp > 0) {
                    sorted_ok = false;
                    printf("Order break in random sample near index %llu: a=(hi=%u, lo=%u), b=(hi=%u, lo=%u)\n",
                           (unsigned long long)i,
                           (unsigned int)a.hi, (unsigned int)a.lo,
                           (unsigned int)b.hi, (unsigned int)b.lo);
                    break;
                }
                if (cmp == 0) duplicate_seen = true;
            }
        }
    }

    fclose(f);

    printf("Sorted check: %s\n", sorted_ok ? "PASS" : "FAIL");
    printf("Duplicate observed: %s\n", duplicate_seen ? "YES" : "NO (in checked scope)");

    if (!sorted_ok) return 3;
    return 0;
}
