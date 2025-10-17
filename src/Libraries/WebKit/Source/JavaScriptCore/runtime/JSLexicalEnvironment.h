/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "CodeBlock.h"
#include "JSSymbolTableObject.h"
#include "SymbolTable.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class LLIntOffsetsExtractor;

class JSLexicalEnvironment : public JSSymbolTableObject {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
public:
    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm)
    {
        static_assert(CellType::needsDestruction == DoesNotNeedDestruction);
        return &vm.variableSizedCellSpace();
    }

    using Base = JSSymbolTableObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnSpecialPropertyNames | OverridesPut;

    WriteBarrierBase<Unknown>* variables()
    {
        return std::bit_cast<WriteBarrierBase<Unknown>*>(std::bit_cast<char*>(this) + offsetOfVariables());
    }

    bool isValidScopeOffset(ScopeOffset offset)
    {
        return !!offset && offset.offset() < symbolTable()->scopeSize();
    }

    WriteBarrierBase<Unknown>& variableAt(ScopeOffset offset)
    {
        ASSERT(isValidScopeOffset(offset));
        return variables()[offset.offset()];
    }

    static size_t offsetOfVariables()
    {
        return WTF::roundUpToMultipleOf<sizeof(WriteBarrier<Unknown>)>(sizeof(JSLexicalEnvironment));
    }

    static size_t offsetOfVariable(ScopeOffset offset)
    {
        Checked<size_t> scopeOffset = offset.offset();
        return offsetOfVariables() + scopeOffset * sizeof(WriteBarrier<Unknown>);
    }

    static size_t allocationSizeForScopeSize(Checked<size_t> scopeSize)
    {
        return offsetOfVariables() + scopeSize * sizeof(WriteBarrier<Unknown>);
    }

    static size_t allocationSize(SymbolTable* symbolTable)
    {
        return allocationSizeForScopeSize(symbolTable->scopeSize());
    }

    static JSLexicalEnvironment* create(
        VM& vm, Structure* structure, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue)
    {
        JSLexicalEnvironment* result =
            new (
                NotNull,
                allocateCell<JSLexicalEnvironment>(vm, allocationSize(symbolTable)))
            JSLexicalEnvironment(vm, structure, currentScope, symbolTable, initialValue);
        result->finishCreation(vm);
        return result;
    }

    static JSLexicalEnvironment* create(VM& vm, JSGlobalObject* globalObject, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue)
    {
        Structure* structure = globalObject->activationStructure();
        return create(vm, structure, currentScope, symbolTable, initialValue);
    }
        
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static void getOwnSpecialPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);

    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);

    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*);

protected:
    JSLexicalEnvironment(VM&, Structure*, JSScope*, SymbolTable*, JSValue initialValue);

    DECLARE_DEFAULT_FINISH_CREATION;

    DECLARE_VISIT_CHILDREN;
    static void analyzeHeap(JSCell*, HeapAnalyzer&);
};

inline JSLexicalEnvironment::JSLexicalEnvironment(VM& vm, Structure* structure, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue)
    : Base(vm, structure, currentScope, symbolTable)
{
    ASSERT(initialValue == jsUndefined() || initialValue == jsTDZValue());
    for (unsigned i = this->symbolTable()->scopeSize(); i--;) {
        // Filling this with undefined/TDZEmptyValue is useful because that's what variables start out as.
        variableAt(ScopeOffset(i)).setStartingValue(initialValue);
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
