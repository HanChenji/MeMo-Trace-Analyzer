#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <sys/types.h>

struct mybitset {
    uint8_t  len;
	uint64_t data;
	mybitset(uint8_t _len) : len(_len), data(0) {
		assert(len <= 64);
	}

    static inline uint64_t bit_mask(size_t pos) { return 1ULL << pos; }
    static inline uint64_t max_mask(size_t len) { return (1ULL << len) - 1; }

	bool get(size_t pos) const {
		return data & bit_mask(pos);
	}

    void set(size_t pos, bool val) {
        data = (data & ~bit_mask(pos)) | bit_mask(pos);
    }

    void operator<<=(size_t shift) {
        data <<= shift;
    }

    uint64_t to_ulong() const {
        return data & max_mask(len);
    }
};
