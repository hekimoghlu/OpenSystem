/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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

#include "B3SwitchCase.h"

namespace JSC { namespace B3 {

class BasicBlock;
class SwitchValue;

// NOTE: You'll always want to include B3CaseCollectionInlines.h when you use this.

class CaseCollection {
public:
    CaseCollection()
    {
    }
    
    CaseCollection(const SwitchValue* terminal, const BasicBlock* owner)
        : m_switch(terminal)
        , m_owner(owner)
    {
    }
    
    const FrequentedBlock& fallThrough() const;

    unsigned size() const;
    SwitchCase at(unsigned index) const;
    
    SwitchCase operator[](unsigned index) const
    {
        return at(index);
    }

    class iterator {
    public:
        iterator()
            : m_collection(nullptr)
            , m_index(0)
        {
        }

        iterator(const CaseCollection& collection, unsigned index)
            : m_collection(&collection)
            , m_index(index)
        {
        }

        SwitchCase operator*()
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
        const CaseCollection* m_collection;
        unsigned m_index;
    };

    typedef iterator const_iterator;

    iterator begin() const { return iterator(*this, 0); }
    iterator end() const { return iterator(*this, size()); }
    
    void dump(PrintStream&) const;
    
private:
    const SwitchValue* m_switch { nullptr };
    const BasicBlock* m_owner { nullptr };
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
