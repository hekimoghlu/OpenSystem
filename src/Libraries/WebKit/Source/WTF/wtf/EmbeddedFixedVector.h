/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#include <iterator>
#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/TrailingArray.h>
#include <wtf/UniqueRef.h>

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(EmbeddedFixedVector);

template<typename T, typename Malloc = EmbeddedFixedVectorMalloc>
class EmbeddedFixedVector final : public TrailingArray<EmbeddedFixedVector<T, Malloc>, T> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(EmbeddedFixedVector);
    WTF_MAKE_NONCOPYABLE(EmbeddedFixedVector);
    WTF_MAKE_NONMOVABLE(EmbeddedFixedVector);
public:
    using Base = TrailingArray<EmbeddedFixedVector<T, Malloc>, T>;

    static UniqueRef<EmbeddedFixedVector> create(unsigned size)
    {
        return UniqueRef { *new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(size) };
    }

    template<typename InputIterator>
    static UniqueRef<EmbeddedFixedVector> create(InputIterator first, InputIterator last)
    {
        unsigned size = Checked<uint32_t> { std::distance(first, last) };
        return UniqueRef { *new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(size, first, last) };
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    static UniqueRef<EmbeddedFixedVector> createFromVector(const Vector<T, inlineCapacity, OverflowHandler>& other)
    {
        unsigned size = Checked<uint32_t> { other.size() }.value();
        return UniqueRef { *new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(size, other.begin(), other.end()) };
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    static UniqueRef<EmbeddedFixedVector> createFromVector(Vector<T, inlineCapacity, OverflowHandler>&& other)
    {
        Vector<T, inlineCapacity, OverflowHandler> container = WTFMove(other);
        unsigned size = Checked<uint32_t> { container.size() }.value();
        return UniqueRef { *new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(size, std::move_iterator { container.begin() }, std::move_iterator { container.end() }) };
    }

    template<typename... Args>
    static UniqueRef<EmbeddedFixedVector> createWithSizeAndConstructorArguments(unsigned size, Args&&... args)
    {
        return UniqueRef { *new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(size, std::forward<Args>(args)...) };
    }

    template<std::invocable<size_t> Generator>
    static std::unique_ptr<EmbeddedFixedVector> createWithSizeFromGenerator(unsigned size, NOESCAPE Generator&& generator)
    {

        auto result = std::unique_ptr<EmbeddedFixedVector> { new (NotNull, Malloc::malloc(Base::allocationSize(size))) EmbeddedFixedVector(typename Base::Failable { }, size, std::forward<Generator>(generator)) };
        if (result->size() != size)
            return nullptr;
        return result;
    }

    UniqueRef<EmbeddedFixedVector> clone() const
    {
        return create(Base::begin(), Base::end());
    }

    bool operator==(const EmbeddedFixedVector& other) const
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
    explicit EmbeddedFixedVector(unsigned size)
        : Base(size)
    {
    }


    template<typename InputIterator>
    EmbeddedFixedVector(unsigned size, InputIterator first, InputIterator last)
        : Base(size, first, last)
    {
    }

    template<typename... Args>
    explicit EmbeddedFixedVector(unsigned size, Args&&... args) // create with given size and constructor arguments for all elements
        : Base(size, std::forward<Args>(args)...)
    {
    }

    template<std::invocable<size_t> FailableGenerator>
    EmbeddedFixedVector(typename Base::Failable failable, unsigned size, FailableGenerator&& generator)
        : Base(failable, size, std::forward<FailableGenerator>(generator))
    {
    }
};

} // namespace WTF

using WTF::EmbeddedFixedVector;
