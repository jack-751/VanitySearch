#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct BloomConfig {
    uint64_t expected_items = 2000000000ULL;
    double fp_rate = 1e-4;
    uint32_t block_bits = 512; // 64 bytes, cache-line friendly
};

struct BloomHeaderV1 {
    char magic[8];             // "BLM2X001"
    uint32_t version;          // 1
    uint32_t hex_len;          // 12..20
    uint64_t expected_items;
    uint64_t inserted_items;
    double fp_rate;
    uint32_t k_hashes;
    uint32_t block_bits;
    uint64_t num_blocks;
    uint64_t seed1;
    uint64_t seed2;
};

class BlockedBloomFilter {
public:
    BlockedBloomFilter() = default;

    bool init(uint32_t hex_len, const BloomConfig &cfg) {
        if (hex_len < 12 || hex_len > 20) return false;
        if (cfg.expected_items == 0) return false;
        if (!(cfg.fp_rate > 0.0 && cfg.fp_rate < 1.0)) return false;
        if (cfg.block_bits == 0 || (cfg.block_bits % 64) != 0) return false;

        header_ = {};
        std::memcpy(header_.magic, "BLM2X001", 8);
        header_.version = 1;
        header_.hex_len = hex_len;
        header_.expected_items = cfg.expected_items;
        header_.inserted_items = 0;
        header_.fp_rate = cfg.fp_rate;
        header_.block_bits = cfg.block_bits;

        const double ln2 = std::log(2.0);
        const double m_bits_f = -1.0 * (double)cfg.expected_items * std::log(cfg.fp_rate) / (ln2 * ln2);
        uint64_t m_bits = (uint64_t)std::ceil(m_bits_f);
        if (m_bits < cfg.block_bits) m_bits = cfg.block_bits;

        uint64_t num_blocks = (m_bits + cfg.block_bits - 1ULL) / cfg.block_bits;
        if (num_blocks == 0) num_blocks = 1;
        header_.num_blocks = num_blocks;

        const double bits_per_item = (double)(num_blocks * cfg.block_bits) / (double)cfg.expected_items;
        uint32_t k = (uint32_t)std::round(bits_per_item * ln2);
        if (k < 4) k = 4;
        if (k > 16) k = 16;
        header_.k_hashes = k;

        header_.seed1 = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)hex_len << 32);
        header_.seed2 = 0xc2b2ae3d27d4eb4fULL ^ ((uint64_t)cfg.block_bits << 16);

        bits_.assign((size_t)(num_blocks * (cfg.block_bits / 64U)), 0ULL);
        return true;
    }

    bool save(const char *path) const {
        if (!path || !path[0]) return false;
        FILE *f = std::fopen(path, "wb");
        if (!f) return false;
        const bool ok =
            (std::fwrite(&header_, sizeof(header_), 1, f) == 1) &&
            (std::fwrite(bits_.data(), sizeof(uint64_t), bits_.size(), f) == bits_.size());
        std::fclose(f);
        return ok;
    }

    bool load(const char *path) {
        if (!path || !path[0]) return false;
        FILE *f = std::fopen(path, "rb");
        if (!f) return false;

        BloomHeaderV1 h{};
        bool ok = std::fread(&h, sizeof(h), 1, f) == 1;
        if (!ok) {
            std::fclose(f);
            return false;
        }
        if (std::memcmp(h.magic, "BLM2X001", 8) != 0 || h.version != 1) {
            std::fclose(f);
            return false;
        }
        if (h.block_bits == 0 || (h.block_bits % 64) != 0 || h.k_hashes == 0 || h.num_blocks == 0) {
            std::fclose(f);
            return false;
        }

        std::vector<uint64_t> bits;
        bits.resize((size_t)(h.num_blocks * (h.block_bits / 64U)));
        if (std::fread(bits.data(), sizeof(uint64_t), bits.size(), f) != bits.size()) {
            std::fclose(f);
            return false;
        }

        std::fclose(f);
        header_ = h;
        bits_.swap(bits);
        return true;
    }

    void add(const uint8_t *data, size_t len) {
        if (!data || len == 0 || bits_.empty()) return;

        const uint64_t h1 = hash64(data, len, header_.seed1);
        const uint64_t h2 = mix64(h1 ^ header_.seed2);
        const uint64_t block_idx = h1 % header_.num_blocks;
        const uint64_t block_words = (uint64_t)(header_.block_bits / 64U);
        const uint64_t base = block_idx * block_words;

        for (uint32_t i = 0; i < header_.k_hashes; i++) {
            const uint64_t x = mix64(h2 + (uint64_t)i * 0x9e3779b97f4a7c15ULL);
            const uint32_t bit = (uint32_t)(x % header_.block_bits);
            bits_[(size_t)(base + (bit >> 6))] |= (1ULL << (bit & 63));
        }

        header_.inserted_items++;
    }

    bool possiblyContains(const uint8_t *data, size_t len) const {
        if (!data || len == 0 || bits_.empty()) return false;

        const uint64_t h1 = hash64(data, len, header_.seed1);
        const uint64_t h2 = mix64(h1 ^ header_.seed2);
        const uint64_t block_idx = h1 % header_.num_blocks;
        const uint64_t block_words = (uint64_t)(header_.block_bits / 64U);
        const uint64_t base = block_idx * block_words;

        for (uint32_t i = 0; i < header_.k_hashes; i++) {
            const uint64_t x = mix64(h2 + (uint64_t)i * 0x9e3779b97f4a7c15ULL);
            const uint32_t bit = (uint32_t)(x % header_.block_bits);
            const uint64_t word = bits_[(size_t)(base + (bit >> 6))];
            if ((word & (1ULL << (bit & 63))) == 0) return false;
        }
        return true;
    }

    uint32_t hexLen() const { return header_.hex_len; }
    uint64_t insertedItems() const { return header_.inserted_items; }
    uint32_t kHashes() const { return header_.k_hashes; }
    uint64_t numBlocks() const { return header_.num_blocks; }

private:
    static uint64_t mix64(uint64_t x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    static uint64_t hash64(const uint8_t *data, size_t len, uint64_t seed) {
        uint64_t h = 1469598103934665603ULL ^ seed;
        for (size_t i = 0; i < len; i++) {
            h ^= (uint64_t)data[i];
            h *= 1099511628211ULL;
        }
        return mix64(h ^ (uint64_t)len);
    }

    BloomHeaderV1 header_{};
    std::vector<uint64_t> bits_;
};
