/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#include "JSSegmentedVariableObject.h"

namespace JSC {

class JSGlobalLexicalEnvironment final : public JSSegmentedVariableObject {
public:
    using Base = JSSegmentedVariableObject;

    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesPut;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.globalLexicalEnvironmentSpace();
    }

    static JSGlobalLexicalEnvironment* create(VM& vm, Structure* structure, JSScope* parentScope)
    {
        JSGlobalLexicalEnvironment* result =
            new (NotNull, allocateCell<JSGlobalLexicalEnvironment>(vm)) JSGlobalLexicalEnvironment(vm, structure, parentScope);
        result->finishCreation(vm);
        result->symbolTable()->setScopeType(SymbolTable::ScopeType::GlobalLexicalScope);
        return result;
    }

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);

    static void destroy(JSCell*);
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    bool isEmpty() const { return !symbolTable()->size(); }
    bool isConstVariable(UniquedStringImpl*);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*);

private:
    JSGlobalLexicalEnvironment(VM& vm, Structure* structure, JSScope* scope)
        : Base(vm, structure, scope)
    {
    }
};

} // namespace JSC
