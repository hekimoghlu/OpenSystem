/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

// This is a generic wrapper around lists of frequented blocks, which gives you just the blocks.

template<typename BasicBlock, typename SuccessorList>
class SuccessorCollection {
public:
    SuccessorCollection(SuccessorList& list)
        : m_list(list)
    {
    }

    size_t size() const { return m_list.size(); }
    BasicBlock* at(size_t index) const { return m_list[index].block(); }
    BasicBlock*& at(size_t index) { return m_list[index].block(); }
    BasicBlock* operator[](size_t index) const { return at(index); }
    BasicBlock*& operator[](size_t index) { return at(index); }

    class iterator {
    public:
        iterator()
            : m_collection(nullptr)
            , m_index(0)
        {
        }

        iterator(SuccessorCollection& collection, size_t index)
            : m_collection(&collection)
            , m_index(index)
        {
        }

        BasicBlock*& operator*() const
        {
            return m_collection->at(m_index);
        }

        iterator& operator++()
        {
            m_index++;
            return *this;
        }

        bool operator==(const iterator& other) const
        {
            ASSERT(m_collection == other.m_collection);
            return m_index == other.m_index;
        }

    private:
        SuccessorCollection* m_collection;
        size_t m_index;
    };

    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, size()); }

    class const_iterator {
    public:
        const_iterator()
            : m_collection(nullptr)
            , m_index(0)
        {
        }

        const_iterator(const SuccessorCollection& collection, size_t index)
            : m_collection(&collection)
            , m_index(index)
        {
        }

        BasicBlock* operator*() const
        {
            return m_collection->at(m_index);
        }

        const_iterator& operator++()
        {
            m_index++;
            return *this;
        }

        bool operator==(const const_iterator& other) const
        {
            ASSERT(m_collection == other.m_collection);
            return m_index == other.m_index;
        }

    private:
        const SuccessorCollection* m_collection;
        size_t m_index;
    };

    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, size()); }

private:
    SuccessorList& m_list;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
