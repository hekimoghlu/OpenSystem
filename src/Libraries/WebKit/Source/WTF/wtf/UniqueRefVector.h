/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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

#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>

namespace WTF {

template<typename T>
class UniqueRefVectorIterator {
    using Iterator = UniqueRefVectorIterator<T>;

public:
    using difference_type = ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    UniqueRefVectorIterator(UniqueRef<T>* iterator)
        : m_iterator(iterator)
    {
    }

    T& operator*() const { return m_iterator->get(); }
    T* operator->() const { return m_iterator->ptr(); }

    friend bool operator==(Iterator, Iterator) = default;

    Iterator& operator++()
    {
        ++m_iterator;
        return *this;
    }
    Iterator& operator--()
    {
        --m_iterator;
        return *this;
    }

private:
    WTF::UniqueRef<T>* m_iterator;
};

template<typename T>
class UniqueRefVectorConstIterator {
    using Iterator = UniqueRefVectorConstIterator<T>;

public:
    using difference_type = ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    UniqueRefVectorConstIterator(const UniqueRef<T>* iterator)
        : m_iterator(iterator)
    {
    }

    const T& operator*() const { return m_iterator->get(); }
    const T* operator->() const { return m_iterator->ptr(); }

    friend bool operator==(Iterator, Iterator) = default;

    Iterator& operator++()
    {
        ++m_iterator;
        return *this;
    }
    Iterator& operator--()
    {
        --m_iterator;
        return *this;
    }

private:
    const WTF::UniqueRef<T>* m_iterator;
};

template<typename T, size_t inlineCapacity = 0>
class UniqueRefVector : public WTF::Vector<WTF::UniqueRef<T>, inlineCapacity> {
    using Base = WTF::Vector<WTF::UniqueRef<T>, inlineCapacity>;

public:
    using ValueType = T;
    using iterator = UniqueRefVectorIterator<T>;
    using const_iterator = UniqueRefVectorConstIterator<T>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    using Base::size;

    iterator begin() { return iterator { Base::begin() }; }
    iterator end() { return iterator { Base::end() }; }
    const_iterator begin() const { return const_iterator { Base::begin() }; }
    const_iterator end() const { return const_iterator { Base::end() }; }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T& at(size_t i) { return Base::at(i).get(); }
    const T& at(size_t i) const { return Base::at(i).get(); }

    T& operator[](size_t i) { return Base::at(i).get(); }
    const T& operator[](size_t i) const { return Base::at(i).get(); }

    T& first() { return Base::at(0).get(); }
    const T& first() const { return Base::at(0).get(); }
    T& last() { return Base::at(Base::size() - 1).get(); }
    const T& last() const { return Base::at(Base::size() - 1).get(); }

    template<typename MatchFunction> size_t findIf(const MatchFunction&) const;
    template<typename MatchFunction> bool containsIf(const MatchFunction& matches) const { return findIf(matches) != notFound; }
};

template<typename T, size_t inlineCapacity>
template<typename MatchFunction>
size_t UniqueRefVector<T, inlineCapacity>::findIf(const MatchFunction& matches) const
{
    for (size_t i = 0; i < size(); ++i) {
        if (matches(at(i)))
            return i;
    }
    return notFound;
}

} // namespace WTF

using WTF::UniqueRefVector;
