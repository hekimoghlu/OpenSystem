/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

#include "JSObject.h"
#include "Options.h"

namespace JSC {

struct DollarVMAssertScope {
    DollarVMAssertScope() { RELEASE_ASSERT(Options::useDollarVM()); }
    ~DollarVMAssertScope() { RELEASE_ASSERT(Options::useDollarVM()); }
};

class JSDollarVM final : public JSNonFinalObject {
public:
    typedef JSNonFinalObject Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertyNames;

    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm)
    {
        return &vm.cellSpace();
    }
    
    DECLARE_EXPORT_INFO;
    
    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        DollarVMAssertScope assertScope;
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

    static JSDollarVM* create(VM& vm, Structure* structure)
    {
        DollarVMAssertScope assertScope;
        JSDollarVM* instance = new (NotNull, allocateCell<JSDollarVM>(vm)) JSDollarVM(vm, structure);
        instance->finishCreation(vm);
        return instance;
    }

    Structure* objectDoingSideEffectPutWithoutCorrectSlotStatusStructure() { return m_objectDoingSideEffectPutWithoutCorrectSlotStatusStructureID.get(); }
    
private:
    JSDollarVM(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
        DollarVMAssertScope assertScope;
    }

    void finishCreation(VM&);
    void addFunction(VM&, JSGlobalObject*, ASCIILiteral name, NativeFunction, unsigned arguments);
    void addConstructibleFunction(VM&, JSGlobalObject*, ASCIILiteral name, NativeFunction, unsigned arguments);

    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);

    DECLARE_VISIT_CHILDREN;

    WriteBarrierStructureID m_objectDoingSideEffectPutWithoutCorrectSlotStatusStructureID;
};

} // namespace JSC
