/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#include <concepts>
#include <type_traits>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// TrailingArray offers the feature trailing array in the derived class.
// We can allocate a memory like the following layout.
//
//     [  DerivedClass  ][ Trailing Array ]
//
// And trailing array offers appropriate methods for accessing and destructions.
template<typename Derived, typename T>
class TrailingArray {
    WTF_MAKE_NONCOPYABLE(TrailingArray);
    friend class JSC::LLIntOffsetsExtractor;
public:
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;
    using const_pointer = const T*;
    using size_type = unsigned;
    using difference_type = std::make_signed_t<size_type>;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

protected:
    explicit TrailingArray(unsigned size)
        : m_size(size)
    {
        static_assert(std::is_final_v<Derived>);
        VectorTypeOperations<T>::initializeIfNonPOD(begin(), end());
    }

    template<typename InputIterator>
    TrailingArray(unsigned size, InputIterator first, InputIterator last)
        : m_size(size)
    {
        static_assert(std::is_final_v<Derived>);
        ASSERT(static_cast<size_t>(std::distance(first, last)) == size);
        std::uninitialized_copy(first, last, begin());
    }

    template<typename... Args>
    TrailingArray(unsigned size, Args&&... args) // create with given size and constructor arguments for all elements
        : m_size(size)
    {
        static_assert(std::is_final_v<Derived>);
        VectorTypeOperations<T>::initializeWithArgs(begin(), end(), std::forward<Args>(args)...);
    }

    // This constructor, which is used via the `Failable` token, will attempt
    // to initialize the array from the generator. The generator returns
    // `std::optional` values, and if one is `nullopt`, that indicates a failure.
    // The constructor sets `m_size` to the index of the most recently successful
    // item to be added in order for the destructor to destroy the right number
    // of elements.
    //
    // It is the responsibility of the caller to check that `size()` is equal
    // to the `size` the caller passed in. If it is not, that is failure, and
    // should be used as appropriate.
    struct Failable { };
    template<std::invocable<size_t> Generator>
    explicit TrailingArray(Failable, unsigned size, NOESCAPE Generator&& generator)
        : m_size(size)
    {
        static_assert(std::is_final_v<Derived>);

        for (size_t i = 0; i < m_size; ++i) {
            if (auto value = generator(i))
                new (NotNull, std::addressof(begin()[i])) T(WTFMove(*value));
            else {
                m_size = i;
                return;
            }
        }
    }

    ~TrailingArray()
    {
        VectorTypeOperations<T>::destruct(begin(), end());
    }

public:
    static constexpr size_t allocationSize(unsigned size)
    {
        return offsetOfData() + size * sizeof(T);
    }

    unsigned size() const { return m_size; }
    bool isEmpty() const { return !size(); }
    unsigned byteSize() const { return size() * sizeof(T); }

    pointer data() LIFETIME_BOUND { return std::bit_cast<T*>(std::bit_cast<uint8_t*>(static_cast<Derived*>(this)) + offsetOfData()); }
    const_pointer data() const LIFETIME_BOUND { return std::bit_cast<const T*>(std::bit_cast<const uint8_t*>(static_cast<const Derived*>(this)) + offsetOfData()); }
    std::span<T> span() LIFETIME_BOUND { return { data(), size() }; }
    std::span<const T> span() const LIFETIME_BOUND { return { data(), size() }; }

    iterator begin() LIFETIME_BOUND { return data(); }
    iterator end() LIFETIME_BOUND { return data() + size(); }
    const_iterator begin() const LIFETIME_BOUND { return cbegin(); }
    const_iterator end() const LIFETIME_BOUND { return cend(); }
    const_iterator cbegin() const LIFETIME_BOUND { return data(); }
    const_iterator cend() const LIFETIME_BOUND { return data() + size(); }

    reverse_iterator rbegin() LIFETIME_BOUND { return reverse_iterator(end()); }
    reverse_iterator rend() LIFETIME_BOUND { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const LIFETIME_BOUND { return crbegin(); }
    const_reverse_iterator rend() const LIFETIME_BOUND { return crend(); }
    const_reverse_iterator crbegin() const LIFETIME_BOUND { return const_reverse_iterator(end()); }
    const_reverse_iterator crend() const LIFETIME_BOUND { return const_reverse_iterator(begin()); }

    reference at(unsigned i) LIFETIME_BOUND
    {
        RELEASE_ASSERT(i < size());
        return begin()[i];
    }

    const_reference at(unsigned i) const LIFETIME_BOUND
    {
        RELEASE_ASSERT(i < size());
        return begin()[i];
    }

    reference operator[](unsigned i) LIFETIME_BOUND { return at(i); }
    const_reference operator[](unsigned i) const LIFETIME_BOUND { return at(i); }

    T& first() LIFETIME_BOUND { return (*this)[0]; }
    const T& first() const LIFETIME_BOUND { return (*this)[0]; }
    T& last() LIFETIME_BOUND { return (*this)[size() - 1]; }
    const T& last() const LIFETIME_BOUND { return (*this)[size() - 1]; }

    void fill(const T& val)
    {
        std::fill(begin(), end(), val);
    }

    static constexpr ptrdiff_t offsetOfSize() { return OBJECT_OFFSETOF(Derived, m_size); }
    static constexpr ptrdiff_t offsetOfData()
    {
        return WTF::roundUpToMultipleOf<alignof(T)>(sizeof(Derived));
    }

protected:
    unsigned m_size { 0 };
};

} // namespace WTF

using WTF::TrailingArray;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
