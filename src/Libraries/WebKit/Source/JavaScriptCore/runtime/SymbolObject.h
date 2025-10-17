/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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

#include "JSWrapperObject.h"
#include "Symbol.h"

namespace JSC {

class SymbolObject final : public JSWrapperObject {
public:
    using Base = JSWrapperObject;

    template<typename, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.symbolObjectSpace<mode>();
    }

    static SymbolObject* create(VM& vm, Structure* structure)
    {
        Symbol* symbol = Symbol::create(vm);
        SymbolObject* object = new (NotNull, allocateCell<SymbolObject>(vm)) SymbolObject(vm, structure);
        object->finishCreation(vm, symbol);
        return object;
    }
    static SymbolObject* create(VM& vm, Structure* structure, Symbol* symbol)
    {
        SymbolObject* object = new (NotNull, allocateCell<SymbolObject>(vm)) SymbolObject(vm, structure);
        object->finishCreation(vm, symbol);
        return object;
    }

    DECLARE_EXPORT_INFO;

    Symbol* internalValue() const { return asSymbol(JSWrapperObject::internalValue()); }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    JS_EXPORT_PRIVATE void finishCreation(VM&, Symbol*);
    JS_EXPORT_PRIVATE SymbolObject(VM&, Structure*);
};
static_assert(sizeof(SymbolObject) == sizeof(JSWrapperObject));

} // namespace JSC
