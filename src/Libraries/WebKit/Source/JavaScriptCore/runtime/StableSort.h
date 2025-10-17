/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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

#include "ArgList.h"
#include <wtf/Int128.h>
#include <wtf/StdLibExtras.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

static ALWAYS_INLINE bool coerceComparatorResultToBoolean(JSGlobalObject* globalObject, JSValue comparatorResult)
{
    if (LIKELY(comparatorResult.isInt32()))
        return comparatorResult.asInt32() < 0;

    // See https://bugs.webkit.org/show_bug.cgi?id=47825 on boolean special-casing
    if (comparatorResult.isBoolean())
        return !comparatorResult.asBoolean();

    return comparatorResult.toNumber(globalObject) < 0;
}

template<typename ElementType, typename Functor>
static ALWAYS_INLINE void arrayInsertionSort(VM& vm, std::span<ElementType> span, const Functor& comparator, size_t sortedHeader = 0)
{
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* array = span.data();
    size_t length = span.size();
    for (size_t i = sortedHeader + 1; i < length; ++i) {
        auto value = array[i];

        // [l, r)
        size_t left = 0;
        size_t right = i;
        for (; left < right;) {
            size_t m = left + (right - left) / 2;
            auto target = array[m];
            bool result = comparator(value, target);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, void());
            if (!result)
                left = m + 1;
            else
                right = m;
        }
        ElementType t = value;
        for (size_t j = left; j < i; ++j)
            std::swap(array[j], t);
        array[i] = t;
    }
}

template<typename ElementType, typename Functor>
static ALWAYS_INLINE void mergePowersortRuns(VM& vm, std::span<ElementType> dst, std::span<const ElementType> src, size_t srcIndex1, size_t srcEnd1, size_t srcIndex2, size_t srcEnd2, const Functor& comparator)
{
    auto scope = DECLARE_THROW_SCOPE(vm);

    size_t left = srcIndex1;
    size_t leftEnd = srcEnd1;
    size_t right = srcIndex2;
    size_t rightEnd = srcEnd2;

    ASSERT(leftEnd <= right);
    ASSERT(rightEnd <= src.size());

    for (size_t dstIndex = left; dstIndex < rightEnd; ++dstIndex) {
        if (right < rightEnd) {
            if (left >= leftEnd) {
                dst[dstIndex] = src[right++];
                continue;
            }
            bool result = comparator(src[right], src[left]);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, void());
            if (result) {
                dst[dstIndex] = src[right++];
                continue;
            }
        }

        dst[dstIndex] = src[left++];
    }
}

// J. Ian Munro and Sebastian Wild. Nearly-Optimal Mergesorts: Fast, Practical Sorting Methods That
// Optimally Adapt to Existing Runs. In 26th Annual European Symposium on Algorithms (ESA 2018).
// Leibniz International Proceedings in Informatics (LIPIcs), Volume 112, pp. 63:1-63:16, Schloss
// Dagstuhl â€“ Leibniz-Zentrum fÃ¼r Informatik (2018) https://doi.org/10.4230/LIPIcs.ESA.2018.63

struct SortedRun {
    size_t m_begin;
    size_t m_end;
};

template<typename ElementType, typename Functor, size_t forceRunLength = 64>
static ALWAYS_INLINE std::span<ElementType> arrayStableSort(VM& vm, std::span<ElementType> src, std::span<ElementType> dst, const Functor& comparator)
{
    constexpr size_t extendRunCutoff = 8;

    auto scope = DECLARE_THROW_SCOPE(vm);

    const size_t numElements = src.size();

    if (!numElements)
        return src;

    // If the array is small, Powersort probably isn't worth it. Just insertion sort.
    if (numElements < extendRunCutoff) {
        scope.release();
        arrayInsertionSort(vm, src.subspan(0, src.size()), comparator);
        return src;
    }

    // power takes in [left, middle-1] and [middle, right]
    auto power = [](size_t left, size_t middle, size_t right, size_t n) -> unsigned {
        UInt128 n1 = middle - left;
        UInt128 n2 = right - middle + 1;
        // a and b are 2*midpoints of the two ranges, so always within [0, 2n)
        UInt128 a = left * 2 + n1;
        UInt128 b = middle * 2 + n2;
        // n <= 2^64, so n << 62 <= 2^126
        // n << 62 must be <= 2^126, so a << 62 must be < 2^127. thus, we don't end up with overflow
        a <<= 62;
        b <<= 62;

        // a is within [0, 2n), so a << 62 is within [0, 2^63 n). Thus, (when we calculate a / n, it must be within [0, 2^63)
        UInt128 differingBits = (a / n) ^ (b / n);
        ASSERT(!(differingBits >> 64));
        return clz(static_cast<uint64_t>(differingBits));
    };

    auto to = dst;
    auto from = src;

    WTF::copyElements(to.data(), spanConstCast<const ElementType>(from).data(), numElements);

    struct PowersortStackEntry {
        SortedRun run;
        unsigned power;
    };

    WTF::Vector<PowersortStackEntry, 64> powerstack;
    // floor(lg(n)) + 1
    powerstack.reserveCapacity(8 * sizeof(numElements) - WTF::clz(numElements));

    SortedRun run1 { 0, 0 };

    // ExtendRunRight(run1.start, n)
    while (run1.m_end + 1 < numElements) {
        bool result = comparator(from[run1.m_end + 1], from[run1.m_end]);
        RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
        if (result)
            break;
        ++run1.m_end;
    }

    if (run1.m_end - run1.m_begin < extendRunCutoff) {
        // If the run is too short, insertion sort a bit
        auto size = std::min(forceRunLength, numElements - run1.m_begin);
        arrayInsertionSort(vm, from.subspan(run1.m_begin, size), comparator, run1.m_end - run1.m_begin);
        RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
        run1.m_end = run1.m_begin + size - 1;
    }

    // See if we can extend the run any more.
    while (run1.m_end + 1 < numElements) {
        bool result = comparator(from[run1.m_end + 1], from[run1.m_end]);
        RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
        if (result)
            break;
        ++run1.m_end;
    }

    while (run1.m_end + 1 < numElements) {
        SortedRun run2 { run1.m_end + 1, run1.m_end + 1 };

        // ExtendRunRight(run2.start, n)
        while (run2.m_end + 1 < numElements) {
            bool result = comparator(from[run2.m_end + 1], from[run2.m_end]);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
            if (result)
                break;
            ++run2.m_end;
        }

        if (run2.m_end - run2.m_begin < extendRunCutoff) {
            // If the run is too short, insertion sort a bit
            auto size = std::min(forceRunLength, numElements - run2.m_begin);
            arrayInsertionSort(vm, from.subspan(run2.m_begin, size), comparator, run2.m_end - run2.m_begin);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
            run2.m_end = run2.m_begin + size - 1;
        }

        // See if we can extend the run any more.
        while (run2.m_end + 1 < numElements) {
            bool result = comparator(from[run2.m_end + 1], from[run2.m_end]);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
            if (result)
                break;
            ++run2.m_end;
        }

        unsigned p = power(run1.m_begin, run2.m_begin, run2.m_end, numElements);
        while (!powerstack.isEmpty() && powerstack.last().power > p) {
            auto rangeToMerge = powerstack.takeLast().run;
            ASSERT(rangeToMerge.m_end == run1.m_begin - 1);

            mergePowersortRuns(vm, to, spanConstCast<const ElementType>(from), rangeToMerge.m_begin, rangeToMerge.m_end + 1, run1.m_begin, run1.m_end + 1, comparator);
            RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
            WTF::copyElements(from.subspan(rangeToMerge.m_begin, run1.m_end + 1 - rangeToMerge.m_begin).data(), spanConstCast<const ElementType>(to).subspan(rangeToMerge.m_begin, run1.m_end + 1 - rangeToMerge.m_begin).data(), run1.m_end + 1 - rangeToMerge.m_begin);
            run1.m_begin = rangeToMerge.m_begin;
        }

        powerstack.append({ run1, p });
        run1 = run2;
    }

    while (!powerstack.isEmpty()) {
        auto rangeToMerge = powerstack.takeLast().run;
        ASSERT(rangeToMerge.m_end == run1.m_begin - 1);

        mergePowersortRuns(vm, to, spanConstCast<const ElementType>(from), rangeToMerge.m_begin, rangeToMerge.m_end + 1, run1.m_begin, run1.m_end + 1, comparator);
        RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope, src);
        WTF::copyElements(from.subspan(rangeToMerge.m_begin, run1.m_end + 1 - rangeToMerge.m_begin).data(), spanConstCast<const ElementType>(to).subspan(rangeToMerge.m_begin, run1.m_end + 1 - rangeToMerge.m_begin).data(), run1.m_end + 1 - rangeToMerge.m_begin);
        run1.m_begin = rangeToMerge.m_begin;
    }

    return (from.data() == src.data()) ? src : dst;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
