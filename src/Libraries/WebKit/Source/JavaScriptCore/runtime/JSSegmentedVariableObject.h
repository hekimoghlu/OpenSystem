/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "ConcurrentJSLock.h"
#include "JSSymbolTableObject.h"
#include "SymbolTable.h"
#include <wtf/SegmentedVector.h>

namespace JSC {

class LLIntOffsetsExtractor;

// This is a mostly drop-in replacement for JSLexicalEnvironment, except that it preserves
// the invariant that after a variable is created, its address in memory will not change
// so long as the JSSegmentedVariableObject is alive. This allows optimizations based
// on getting the address of the variable and remembering it. As well, unlike a
// JSLexicalEnvironment, this will manage the memory for the registers itself and neither
// requires nor allows for the subclasses to manage that memory. Finally,
// JSSegmentedVariableObject has its own GC tracing functionality, since it knows the
// exact dimensions of the variables array at all times.

class JSSegmentedVariableObject : public JSSymbolTableObject {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
public:
    using Base = JSSymbolTableObject;

    DECLARE_INFO;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    template<typename CellType, SubspaceAccess>
    static void subspaceFor(VM&) { RELEASE_ASSERT_NOT_REACHED(); }

    bool isValidScopeOffset(ScopeOffset offset)
    {
        return !!offset && offset.offset() < m_variables.size();
    }

    // This is not thread-safe, since m_variables is a segmented vector, and its spine can resize with
    // malloc/free if new variables - unrelated to the one you are accessing - are added. You can get
    // around this by grabbing m_lock, or finding some other way to get to the variable pointer (global
    // variable access bytecode instructions will have a direct pointer already).
    WriteBarrier<Unknown>& variableAt(ScopeOffset offset) { return m_variables[offset.offset()]; }
    
    // This is a slow method call, which searches the register bank to find the index
    // given a pointer. It will CRASH() if it does not find the register. Only use this
    // in debug code (like bytecode dumping).
    JS_EXPORT_PRIVATE ScopeOffset findVariableIndex(void*);
    
    WriteBarrier<Unknown>* assertVariableIsInThisObject(WriteBarrier<Unknown>* variablePointer)
    {
        if (ASSERT_ENABLED)
            findVariableIndex(variablePointer);
        return variablePointer;
    }
    
    // Adds numberOfRegistersToAdd registers, initializes them to Undefined, and returns
    // the index of the first one added.
    JS_EXPORT_PRIVATE ScopeOffset addVariables(unsigned numberOfVariablesToAdd, JSValue);

    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE);
    JS_EXPORT_PRIVATE static void analyzeHeap(JSCell*, HeapAnalyzer&);
    
protected:
    JSSegmentedVariableObject(VM&, Structure*, JSScope*);
    
    ~JSSegmentedVariableObject();

    void finishCreation(VM&);
    
private:
    SegmentedVector<WriteBarrier<Unknown>, 16> m_variables;
#ifndef NDEBUG
    bool m_alreadyDestroyed { false }; // We use these assertions to check that we aren't doing ancient hacks that result in this being destroyed more than once.
#endif
};

} // namespace JSC
