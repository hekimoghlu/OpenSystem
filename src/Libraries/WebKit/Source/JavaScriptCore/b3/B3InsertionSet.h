/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#include "B3Origin.h"
#include "B3Type.h"
#include "B3TypeMap.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class BasicBlock;
class Procedure;
class Value;

typedef WTF::Insertion<Value*> Insertion;

class InsertionSet {
public:
    InsertionSet(Procedure& procedure)
        : m_procedure(procedure)
    {
    }

    bool isEmpty() const { return m_insertions.isEmpty(); }

    Procedure& code() { return m_procedure; }

    void appendInsertion(const Insertion& insertion)
    {
        m_insertions.append(insertion);
    }

    Value* insertValue(size_t index, Value* value)
    {
        appendInsertion(Insertion(index, value));
        return value;
    }

    template<typename ValueType, typename... Arguments>
    ValueType* insert(size_t index, Arguments... arguments);

    Value* insertIntConstant(size_t index, Origin, Type, int64_t value);
    Value* insertIntConstant(size_t index, Value* likeValue, int64_t value);

    Value* insertBottom(size_t index, Origin, Type);
    Value* insertBottom(size_t index, Value*);
    
    Value* insertClone(size_t index, Value*);

    void execute(BasicBlock*);

private:
    Procedure& m_procedure;
    Vector<Insertion, 8> m_insertions;

    TypeMap<Value*> m_bottomForType;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
