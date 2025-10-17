/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

#include "AirInsertionSet.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 { namespace Air {

class BasicBlock;
class Code;

// Phased insertions allow you to ascribe phases to the things inserted at an instruction boundary.
class PhaseInsertion : public Insertion {
public:
    PhaseInsertion() { }
    
    template<typename T>
    PhaseInsertion(size_t index, unsigned phase, T&& element)
        : Insertion(index, std::forward<T>(element))
        , m_phase(phase)
    {
    }
    
    unsigned phase() const { return m_phase; }
    
    bool operator<(const PhaseInsertion& other) const
    {
        if (index() != other.index())
            return index() < other.index();
        return m_phase < other.m_phase;
    }

private:
    unsigned m_phase { 0 };
};

class PhaseInsertionSet {
public:
    PhaseInsertionSet()
    {
    }
    
    template<typename T>
    void appendInsertion(T&& insertion)
    {
        m_insertions.append(std::forward<T>(insertion));
    }
    
    template<typename Inst>
    void insertInst(size_t index, unsigned phase, Inst&& inst)
    {
        appendInsertion(PhaseInsertion(index, phase, std::forward<Inst>(inst)));
    }
    
    template<typename... Arguments>
    void insert(size_t index, unsigned phase, Arguments&&... arguments)
    {
        insertInst(index, phase, Inst(std::forward<Arguments>(arguments)...));
    }
    
    void execute(BasicBlock*);

private:
    Vector<PhaseInsertion, 8> m_insertions;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

