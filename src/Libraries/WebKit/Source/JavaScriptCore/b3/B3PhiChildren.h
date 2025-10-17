/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#include "B3Procedure.h"
#include "B3UpsilonValue.h"
#include <wtf/GraphNodeWorklist.h>
#include <wtf/IndexMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

class PhiChildren {
    WTF_MAKE_TZONE_ALLOCATED(PhiChildren);
public:
    PhiChildren(Procedure&);
    ~PhiChildren();

    class ValueCollection {
    public:
        ValueCollection(Vector<UpsilonValue*>* values = nullptr)
            : m_values(values)
        {
        }

        unsigned size() const { return m_values->size(); }
        Value* at(unsigned index) const { return m_values->at(index)->child(0); }
        Value* operator[](unsigned index) const { return at(index); }

        bool contains(Value* value) const
        {
            for (unsigned i = size(); i--;) {
                if (at(i) == value)
                    return true;
            }
            return false;
        }

        class iterator {
        public:
            iterator(Vector<UpsilonValue*>* values = nullptr, unsigned index = 0)
                : m_values(values)
                , m_index(index)
            {
            }

            Value* operator*() const
            {
                return m_values->at(m_index)->child(0);
            }

            iterator& operator++()
            {
                m_index++;
                return *this;
            }

            bool operator==(const iterator& other) const
            {
                ASSERT(m_values == other.m_values);
                return m_index == other.m_index;
            }

        private:
            Vector<UpsilonValue*>* m_values;
            unsigned m_index;
        };

        iterator begin() const { return iterator(m_values); }
        iterator end() const { return iterator(m_values, m_values->size()); }

    private:
        Vector<UpsilonValue*>* m_values;
    };
    
    class UpsilonCollection {
    public:
        UpsilonCollection()
        {
        }
        
        UpsilonCollection(PhiChildren* phiChildren, Value* value, Vector<UpsilonValue*>* values)
            : m_phiChildren(phiChildren)
            , m_value(value)
            , m_values(values)
        {
        }

        unsigned size() const { return m_values->size(); }
        Value* at(unsigned index) const { return m_values->at(index); }
        Value* operator[](unsigned index) const { return at(index); }

        bool contains(Value* value) const { return m_values->contains(value); }

        typedef Vector<UpsilonValue*>::const_iterator iterator;
        Vector<UpsilonValue*>::const_iterator begin() const { return m_values->begin(); }
        Vector<UpsilonValue*>::const_iterator end() const { return m_values->end(); }

        ValueCollection values() { return ValueCollection(m_values); }
        
        template<typename Functor>
        void forAllTransitiveIncomingValues(const Functor& functor)
        {
            if (m_value->opcode() != Phi) {
                functor(m_value);
                return;
            }
            
            GraphNodeWorklist<Value*> worklist;
            worklist.push(m_value);
            while (Value* phi = worklist.pop()) {
                for (Value* child : m_phiChildren->at(phi).values()) {
                    if (child->opcode() == Phi)
                        worklist.push(child);
                    else
                        functor(child);
                }
            }
        }

        bool transitivelyUses(Value* candidate)
        {
            bool result = false;
            forAllTransitiveIncomingValues(
                [&] (Value* child) {
                    result |= child == candidate;
                });
            return result;
        }

    private:
        PhiChildren* m_phiChildren { nullptr };
        Value* m_value { nullptr };
        Vector<UpsilonValue*>* m_values { nullptr };
    };

    UpsilonCollection at(Value* value) { return UpsilonCollection(this, value, &m_upsilons[value]); }
    UpsilonCollection operator[](Value* value) { return at(value); }

    const Vector<Value*, 8>& phis() const { return m_phis; }

private:
    IndexMap<Value*, Vector<UpsilonValue*>> m_upsilons;
    Vector<Value*, 8> m_phis;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
