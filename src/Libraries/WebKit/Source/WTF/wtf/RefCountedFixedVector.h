/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

#include <type_traits>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/TrailingArray.h>

namespace WTF {

template<typename T, bool isThreadSafe>
class RefCountedFixedVectorBase final : public std::conditional<isThreadSafe, ThreadSafeRefCounted<RefCountedFixedVectorBase<T, isThreadSafe>>, RefCounted<RefCountedFixedVectorBase<T, isThreadSafe>>>::type, public TrailingArray<RefCountedFixedVectorBase<T, isThreadSafe>, T> {
public:
    using Base = TrailingArray<RefCountedFixedVectorBase<T, isThreadSafe>, T>;

    static Ref<RefCountedFixedVectorBase> create(unsigned size)
    {
        return adoptRef(*new (NotNull, fastMalloc(Base::allocationSize(size))) RefCountedFixedVectorBase(size));
    }

    template<typename InputIterator>
    static Ref<RefCountedFixedVectorBase> create(InputIterator first, InputIterator last)
    {
        unsigned size = Checked<uint32_t> { std::distance(first, last) };
        return adoptRef(*new (NotNull, fastMalloc(Base::allocationSize(size))) RefCountedFixedVectorBase(size, first, last));
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    static Ref<RefCountedFixedVectorBase> createFromVector(const Vector<T, inlineCapacity, OverflowHandler>& other)
    {
        unsigned size = Checked<uint32_t> { other.size() }.value();
        return adoptRef(*new (NotNull, fastMalloc(Base::allocationSize(size))) RefCountedFixedVectorBase(size, std::begin(other), std::end(other)));
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    static Ref<RefCountedFixedVectorBase> createFromVector(Vector<T, inlineCapacity, OverflowHandler>&& other)
    {
        Vector<T, inlineCapacity, OverflowHandler> container = WTFMove(other);
        unsigned size = Checked<uint32_t> { container.size() }.value();
        return adoptRef(*new (NotNull, fastMalloc(Base::allocationSize(size))) RefCountedFixedVectorBase(size, std::move_iterator { container.begin() }, std::move_iterator { container.end() }));
    }

    Ref<RefCountedFixedVectorBase> clone() const
    {
        return create(Base::begin(), Base::end());
    }

    bool operator==(const RefCountedFixedVectorBase& other) const
    {
        if (Base::size() != other.size())
            return false;
        for (unsigned i = 0; i < Base::size(); ++i) {
            if (Base::at(i) != other.at(i))
                return false;
        }
        return true;
    }

private:
    explicit RefCountedFixedVectorBase(unsigned size)
        : Base(size)
    {
    }

    template<typename InputIterator>
    RefCountedFixedVectorBase(unsigned size, InputIterator first, InputIterator last)
        : Base(size, first, last)
    {
    }
};

template<typename T>
using RefCountedFixedVector = RefCountedFixedVectorBase<T, false>;
template<typename T>
using ThreadSafeRefCountedFixedVector = RefCountedFixedVectorBase<T, true>;

} // namespace WTF

using WTF::RefCountedFixedVector;
using WTF::ThreadSafeRefCountedFixedVector;
