/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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

#include "JSLexicalEnvironment.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class AbstractModuleRecord;
class Register;

class JSModuleEnvironment final : public JSLexicalEnvironment {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
public:
    using Base = JSLexicalEnvironment;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnSpecialPropertyNames | OverridesPut;

    static JSModuleEnvironment* create(VM& vm, JSGlobalObject* globalObject, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue, AbstractModuleRecord* moduleRecord)
    {
        Structure* structure = globalObject->moduleEnvironmentStructure();
        return create(vm, structure, currentScope, symbolTable, initialValue, moduleRecord);
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*);

    static size_t offsetOfModuleRecord(SymbolTable* symbolTable)
    {
        size_t offset = Base::allocationSize(symbolTable);
        ASSERT(WTF::roundUpToMultipleOf<sizeof(WriteBarrier<AbstractModuleRecord>)>(offset) == offset);
        return offset;
    }

    static size_t allocationSize(SymbolTable* symbolTable)
    {
        return offsetOfModuleRecord(symbolTable) + sizeof(WriteBarrier<AbstractModuleRecord>);
    }

    AbstractModuleRecord* moduleRecord()
    {
        return moduleRecordSlot().get();
    }

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static void getOwnSpecialPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);

private:
    JSModuleEnvironment(VM&, Structure*, JSScope*, SymbolTable*, JSValue initialValue, AbstractModuleRecord*);

    static JSModuleEnvironment* create(VM&, Structure*, JSScope*, SymbolTable*, JSValue initialValue, AbstractModuleRecord*);

    DECLARE_DEFAULT_FINISH_CREATION;

    WriteBarrierBase<AbstractModuleRecord>& moduleRecordSlot()
    {
        return *std::bit_cast<WriteBarrierBase<AbstractModuleRecord>*>(std::bit_cast<char*>(this) + offsetOfModuleRecord(symbolTable()));
    }

    DECLARE_VISIT_CHILDREN;
};

inline JSModuleEnvironment::JSModuleEnvironment(VM& vm, Structure* structure, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue, AbstractModuleRecord* moduleRecord)
    : Base(vm, structure, currentScope, symbolTable, initialValue)
{
    this->moduleRecordSlot().setWithoutWriteBarrier(moduleRecord);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
