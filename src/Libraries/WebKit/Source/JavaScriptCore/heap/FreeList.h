/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#pragma once

#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/StdLibExtras.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class HeapCell;

struct FreeCell {
    static ALWAYS_INLINE uint64_t scramble(int32_t offsetToNext, uint32_t lengthInBytes, uint64_t secret)
    {
        ASSERT(static_cast<uint64_t>(lengthInBytes) << 32 | offsetToNext);
        return (static_cast<uint64_t>(lengthInBytes) << 32 | offsetToNext) ^ secret;
    }

    static ALWAYS_INLINE std::tuple<int32_t, uint32_t> descramble(uint64_t scrambledBits, uint64_t secret)
    {
        static_assert(WTF::isPowerOfTwo(sizeof(FreeCell))); // Make sure this division isn't super costly.
        uint64_t descrambledBits = scrambledBits ^ secret;
        return { static_cast<int32_t>(static_cast<uint32_t>(descrambledBits)), static_cast<uint32_t>(descrambledBits >> 32u) };
    }

    ALWAYS_INLINE void makeLast(uint32_t lengthInBytes, uint64_t secret)
    {
        scrambledBits = scramble(1, lengthInBytes, secret); // We use a set LSB to indicate a sentinel pointer.
    }

    ALWAYS_INLINE void setNext(FreeCell* next, uint32_t lengthInBytes, uint64_t secret)
    {
        scrambledBits = scramble((next - this) * sizeof(FreeCell), lengthInBytes, secret);
    }

    ALWAYS_INLINE std::tuple<int32_t, uint32_t> decode(uint64_t secret)
    {
        return descramble(scrambledBits, secret);
    }

    static ALWAYS_INLINE void advance(uint64_t secret, FreeCell*& interval, char*& intervalStart, char*& intervalEnd)
    {
        auto [offsetToNext, lengthInBytes] = interval->decode(secret);
        intervalStart = std::bit_cast<char*>(interval);
        intervalEnd = intervalStart + lengthInBytes;
        interval = std::bit_cast<FreeCell*>(intervalStart + offsetToNext);
    }

    static constexpr ptrdiff_t offsetOfScrambledBits() { return OBJECT_OFFSETOF(FreeCell, scrambledBits); }

    uint64_t preservedBitsForCrashAnalysis;
    uint64_t scrambledBits;
};

class FreeList {
public:
    FreeList(unsigned cellSize);
    ~FreeList();
    
    void clear();
    
    JS_EXPORT_PRIVATE void initialize(FreeCell* head, uint64_t secret, unsigned bytes);
    
    bool allocationWillFail() const { return m_intervalStart >= m_intervalEnd && isSentinel(nextInterval()); }
    bool allocationWillSucceed() const { return !allocationWillFail(); }
    
    template<typename Func>
    HeapCell* allocateWithCellSize(const Func& slowPath, size_t cellSize);
    
    template<typename Func>
    void forEach(const Func&) const;
    
    unsigned originalSize() const { return m_originalSize; }

    static bool isSentinel(FreeCell* cell) { return std::bit_cast<uintptr_t>(cell) & 1; }
    static constexpr ptrdiff_t offsetOfNextInterval() { return OBJECT_OFFSETOF(FreeList, m_nextInterval); }
    static constexpr ptrdiff_t offsetOfSecret() { return OBJECT_OFFSETOF(FreeList, m_secret); }
    static constexpr ptrdiff_t offsetOfIntervalStart() { return OBJECT_OFFSETOF(FreeList, m_intervalStart); }
    static constexpr ptrdiff_t offsetOfIntervalEnd() { return OBJECT_OFFSETOF(FreeList, m_intervalEnd); }
    static constexpr ptrdiff_t offsetOfOriginalSize() { return OBJECT_OFFSETOF(FreeList, m_originalSize); }
    static constexpr ptrdiff_t offsetOfCellSize() { return OBJECT_OFFSETOF(FreeList, m_cellSize); }
    
    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

    unsigned cellSize() const { return m_cellSize; }
    
private:
    FreeCell* nextInterval() const { return m_nextInterval; }
    
    char* m_intervalStart { nullptr };
    char* m_intervalEnd { nullptr };
    FreeCell* m_nextInterval { std::bit_cast<FreeCell*>(static_cast<uintptr_t>(1)) };
    uint64_t m_secret { 0 };
    unsigned m_originalSize { 0 };
    unsigned m_cellSize { 0 };
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
