/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef Bitmap_h
#define Bitmap_h

#include <bit>
#include <span>

#include "Allocator.h"

//FIXME: Should we get rid of the size tracking and make this dynamically grow as necessary?

namespace lsl {

struct VIS_HIDDEN Bitmap {
    Bitmap() = default;
    Bitmap(Allocator& allocator, size_t size) : _allocator(&allocator), _size(size) {
        _sizeInBytes = (_size + (kBitsPerByte-1)) / kBitsPerByte;
        void* buffer = _allocator->malloc(_sizeInBytes);
        bzero(buffer, _sizeInBytes);
        _bitmap = UniquePtr<std::byte>((std::byte*)buffer);
    }
    Bitmap(Allocator& allocator, size_t size, std::span<std::byte>& data) : Bitmap(allocator, size) {
        _bitmap.withUnsafe([&](std::byte* bitmap) {
            std::copy(data.begin(), data.begin()+_sizeInBytes, &bitmap[0]);
        });
        data = data.last((size_t)(data.size()-_sizeInBytes));
    }
    Bitmap(const Bitmap& other) : Bitmap(*other._allocator, other._size) {
        _bitmap.withUnsafe([&](std::byte* bitmap) {
            std::copy(other.bytes().begin(), other.bytes().end(), &bitmap[0]);
        });
    }
    Bitmap& operator=(const Bitmap& other) {
        auto temp = other;
        swap(temp);
        return *this;
    }
    Bitmap(Bitmap&& other) {
        swap(other);
    }
    Bitmap& operator=(Bitmap&& other) {
        swap(other);
        return *this;
    }
    ~Bitmap() {
        clear();
    }
    void setBit(size_t bit) {
        assert(bit < _size);
        _bitmap.withUnsafe([&](std::byte* bitmap) {
            std::byte* byte = &bitmap[bit / kBitsPerByte];
            *byte |= (std::byte)(1 << (bit % kBitsPerByte));
        });
    }
    bool checkBit(size_t bit) const {
        assert(bit < _size);
        return ((std::byte)0 != _bitmap.withUnsafe([&](std::byte* bitmap) {
            std::byte* byte = &bitmap[bit / kBitsPerByte];
            return *byte & (std::byte)(1 << (bit % kBitsPerByte));
        }));
    }
    size_t size() const {
        return _size;
    }
    size_t sizeInBytes() const {
        return _sizeInBytes;
    }
    friend void swap(Bitmap& x, Bitmap& y) {
        x.swap(y);
    }
    std::span<std::byte> bytes() const {
        return _bitmap.withUnsafe([&](std::byte* bitmap) {
            return std::span<std::byte>(&bitmap[0], sizeInBytes());
        });
    }
    void clear() {
        _size   = 0;
        _sizeInBytes = 0;
        _bitmap = nullptr;
    }
    size_t getBitCount() const {
        return _bitmap.withUnsafe([&](std::byte* bitmap) {
            uint64_t wordCount = roundUpToPowerOf2<kBitsPerWord>(_size) / kBitsPerWord;
            size_t result = 0;
            for (auto i = 0; i < wordCount; ++i) {
                result += std::popcount(((uint64_t*)bitmap)[i]);
            }
            return result;
        });
    }
    explicit operator bool() const {
        return (bool)_bitmap;
    }
private:
    static const uint64_t kBitsPerByte = 8;
    static const uint64_t kBitsPerWord = 64;

    template<uint64_t n>
    static inline uint64_t roundUpToPowerOf2(uint64_t value) {
        static_assert(std::popcount(n) == 1 && "Can only align to powers of 2");
      return ((value + (n-1)) & (-1*n));
    }

    void swap(Bitmap& other) {
        if (this == &other) { return; }
        using std::swap;
        swap(_allocator,    other._allocator);
        swap(_bitmap,       other._bitmap);
        swap(_size,         other._size);
        swap(_sizeInBytes,  other._sizeInBytes);
    }
    Allocator*              _allocator   = nullptr;
    UniquePtr<std::byte>    _bitmap      = nullptr;
    size_t                  _size        = 0;
    size_t                  _sizeInBytes = 0;
};

};

#endif /* Bitmap_h */
