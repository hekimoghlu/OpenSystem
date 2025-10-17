/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

namespace WTF {

    template<typename KeyTypeArg, typename ValueTypeArg>
    struct KeyValuePair;

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableConstKeysIterator;
    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableConstValuesIterator;
    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableKeysIterator;
    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableValuesIterator;

    template<typename HashTableType, typename ValueType> struct HashTableConstIteratorAdapter;
    template<typename HashTableType, typename ValueType> struct HashTableIteratorAdapter;

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableConstIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KeyValuePair<KeyType, MappedType>;
        using difference_type = ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        typedef KeyValuePair<KeyType, MappedType> ValueType;
    public:
        typedef HashTableConstKeysIterator<HashTableType, KeyType, MappedType> Keys;
        typedef HashTableConstValuesIterator<HashTableType, KeyType, MappedType> Values;

        HashTableConstIteratorAdapter() {}
        HashTableConstIteratorAdapter(const typename HashTableType::const_iterator& impl) : m_impl(impl) {}

        const ValueType* get() const { return (const ValueType*)m_impl.get(); }
        const ValueType& operator*() const { return *get(); }
        const ValueType* operator->() const { return get(); }

        HashTableConstIteratorAdapter& operator++() { ++m_impl; return *this; }
        HashTableConstIteratorAdapter& operator++(int) { auto result = *this; ++m_impl; return result; }

        Keys keys() { return Keys(*this); }
        Values values() { return Values(*this); }

        typename HashTableType::const_iterator m_impl;
    };

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KeyValuePair<KeyType, MappedType>;
        using difference_type = ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

    private:
        typedef KeyValuePair<KeyType, MappedType> ValueType;
    public:
        typedef HashTableKeysIterator<HashTableType, KeyType, MappedType> Keys;
        typedef HashTableValuesIterator<HashTableType, KeyType, MappedType> Values;

        HashTableIteratorAdapter() {}
        HashTableIteratorAdapter(const typename HashTableType::iterator& impl) : m_impl(impl) {}

        ValueType* get() const { return (ValueType*)m_impl.get(); }
        ValueType& operator*() const { return *get(); }
        ValueType* operator->() const { return get(); }

        HashTableIteratorAdapter& operator++() { ++m_impl; return *this; }
        HashTableIteratorAdapter& operator++(int) { auto result = *this; ++m_impl; return result; }

        operator HashTableConstIteratorAdapter<HashTableType, ValueType>() {
            typename HashTableType::const_iterator i = m_impl;
            return i;
        }

        Keys keys() { return Keys(*this); }
        Values values() { return Values(*this); }

        typename HashTableType::iterator m_impl;
    };

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableConstKeysIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KeyType;
        using difference_type = ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        typedef HashTableConstIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> ConstIterator;

    public:
        HashTableConstKeysIterator() { }
        HashTableConstKeysIterator(const ConstIterator& impl) : m_impl(impl) {}
        
        const KeyType* get() const { return &(m_impl.get()->key); }
        const KeyType& operator*() const { return *get(); }
        const KeyType* operator->() const { return get(); }

        HashTableConstKeysIterator& operator++() { ++m_impl; return *this; }
        HashTableConstKeysIterator& operator++(int) { auto result = *this; ++m_impl; return result; }

        ConstIterator m_impl;
    };

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableConstValuesIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = MappedType;
        using difference_type = ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        typedef HashTableConstIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> ConstIterator;

    public:
        HashTableConstValuesIterator() { }
        HashTableConstValuesIterator(const ConstIterator& impl) : m_impl(impl) {}
        
        const MappedType* get() const { return std::addressof(m_impl.get()->value); }
        const MappedType& operator*() const { return *get(); }
        const MappedType* operator->() const { return get(); }

        HashTableConstValuesIterator& operator++() { ++m_impl; return *this; }
        HashTableConstValuesIterator& operator++(int) { auto result = *this; ++m_impl; return result; }

        ConstIterator m_impl;
    };

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableKeysIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KeyType;
        using difference_type = ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

    private:
        typedef HashTableIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> Iterator;
        typedef HashTableConstIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> ConstIterator;

    public:
        HashTableKeysIterator() { }
        HashTableKeysIterator(const Iterator& impl) : m_impl(impl) {}
        
        KeyType* get() const { return &(m_impl.get()->key); }
        KeyType& operator*() const { return *get(); }
        KeyType* operator->() const { return get(); }

        HashTableKeysIterator& operator++() { ++m_impl; return *this; }
        HashTableKeysIterator& operator++(int) { auto result = *this; ++m_impl; return result; }

        operator HashTableConstKeysIterator<HashTableType, KeyType, MappedType>() {
            ConstIterator i = m_impl;
            return i;
        }

        Iterator m_impl;
    };

    template<typename HashTableType, typename KeyType, typename MappedType> struct HashTableValuesIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = MappedType;
        using difference_type = ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

    private:
        typedef HashTableIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> Iterator;
        typedef HashTableConstIteratorAdapter<HashTableType, KeyValuePair<KeyType, MappedType>> ConstIterator;

    public:
        HashTableValuesIterator() { }
        HashTableValuesIterator(const Iterator& impl) : m_impl(impl) {}
        
        MappedType* get() const { return std::addressof(m_impl.get()->value); }
        MappedType& operator*() const { return *get(); }
        MappedType* operator->() const { return get(); }

        HashTableValuesIterator& operator++() { ++m_impl; return *this; }
        HashTableValuesIterator& operator++(int) { auto result = *this; ++m_impl; return result; }

        operator HashTableConstValuesIterator<HashTableType, KeyType, MappedType>() {
            ConstIterator i = m_impl;
            return i;
        }

        Iterator m_impl;
    };

    template<typename T, typename U, typename V>
        inline bool operator==(const HashTableConstKeysIterator<T, U, V>& a, const HashTableConstKeysIterator<T, U, V>& b)
    {
        return a.m_impl == b.m_impl;
    }

    template<typename T, typename U, typename V>
        inline bool operator==(const HashTableConstValuesIterator<T, U, V>& a, const HashTableConstValuesIterator<T, U, V>& b)
    {
        return a.m_impl == b.m_impl;
    }

    template<typename T, typename U, typename V>
        inline bool operator==(const HashTableKeysIterator<T, U, V>& a, const HashTableKeysIterator<T, U, V>& b)
    {
        return a.m_impl == b.m_impl;
    }

    template<typename T, typename U, typename V>
        inline bool operator==(const HashTableValuesIterator<T, U, V>& a, const HashTableValuesIterator<T, U, V>& b)
    {
        return a.m_impl == b.m_impl;
    }

} // namespace WTF
