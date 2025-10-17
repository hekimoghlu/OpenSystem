/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#ifndef AudioArray_h
#define AudioArray_h

#include <span>
#include <string.h>
#include <wtf/CheckedArithmetic.h>
#include <wtf/MallocSpan.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

template<typename T>
class AudioArray {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(AudioArray);
public:
    AudioArray() = default;
    explicit AudioArray(size_t n)
    {
        resize(n);
    }

    ~AudioArray() = default;

    // It's OK to call resize() multiple times, but data will *not* be copied from an initial allocation
    // if re-allocated. Allocations are zero-initialized.
    void resize(Checked<size_t> n)
    {
        if (n == size())
            return;

        Checked<size_t> initialSize = sizeof(T) * n;
        // Accelerate.framework behaves differently based on input vector alignment. And each implementation
        // has very small difference in output! We ensure 32byte alignment so that we will always take the most
        // optimized implementation if possible, which makes the result deterministic.
        constexpr size_t alignment = 32;

        m_allocation = MallocSpan<T, FastAlignedMalloc>::alignedMalloc(alignment, initialSize);
        zero();
    }

    std::span<T> span() { return m_allocation.mutableSpan(); }
    std::span<const T> span() const { return m_allocation.span(); }
    T* data() { return span().data(); }
    const T* data() const { return span().data(); }
    size_t size() const { return span().size(); }
    bool isEmpty() const { return span().empty(); }

    T& at(size_t i) { return m_allocation[i]; }
    const T& at(size_t i) const { return m_allocation[i]; }
    T& operator[](size_t i) { return at(i); }
    const T& operator[](size_t i) const { return at(i); }

    void zero() { zeroSpan(span()); }

    void zeroRange(unsigned start, unsigned end)
    {
        bool isSafe = (start <= end) && (end <= size());
        ASSERT(isSafe);
        if (!isSafe)
            return;

        // This expression cannot overflow because end - start cannot be
        // greater than m_size, which is safe due to the check in resize().
        zeroSpan(span().subspan(start, end - start));
    }

    void copyToRange(std::span<const T> sourceData, unsigned start, unsigned end)
    {
        bool isSafe = (start <= end) && (end <= size());
        ASSERT(isSafe);
        if (!isSafe)
            return;

        // This expression cannot overflow because end - start cannot be
        // greater than m_size, which is safe due to the check in resize().
        memcpySpan(this->span().subspan(start), sourceData.first(end - start));
    }

    bool containsConstantValue() const
    {
        if (size() <= 1)
            return true;
        float constant = m_allocation[0];
        for (auto& value : span().subspan(1)) {
            if (value != constant)
                return false;
        }
        return true;
    }

private:
    MallocSpan<T, FastAlignedMalloc> m_allocation;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename T>, AudioArray<T>);

typedef AudioArray<float> AudioFloatArray;
typedef AudioArray<double> AudioDoubleArray;

} // WebCore

#endif // AudioArray_h
