/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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

#include "AirInst.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 { namespace Air {

class BasicBlock;
class Code;

typedef WTF::Insertion<Inst> Insertion;

class InsertionSet {
public:
    InsertionSet(Code& code)
        : m_code(&code)
    {
    }

    Code& code() { return *m_code; }

    template<typename T>
    void appendInsertion(T&& insertion)
    {
        m_insertions.append(std::forward<T>(insertion));
    }

    template<typename Inst>
    void insertInst(size_t index, Inst&& inst)
    {
        appendInsertion(Insertion(index, std::forward<Inst>(inst)));
    }

    template<typename InstVector>
    void insertInsts(size_t index, const InstVector& insts)
    {
        for (const Inst& inst : insts)
            insertInst(index, inst);
    }
    void insertInsts(size_t index, Vector<Inst>&&);
    
    template<typename... Arguments>
    void insert(size_t index, Arguments&&... arguments)
    {
        insertInst(index, Inst(std::forward<Arguments>(arguments)...));
    }

    void execute(BasicBlock*);

private:
    Code* m_code; // Pointer so that this can be copied.
    Vector<Insertion, 8> m_insertions;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
