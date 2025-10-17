/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

namespace JSC {

class ObjectPrototype final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | IsImmutablePrototypeExoticObject;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(ObjectPrototype, Base);
        return &vm.plainObjectSpace();
    }

    static ObjectPrototype* create(VM&, JSGlobalObject*, Structure*);

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    ObjectPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JS_EXPORT_PRIVATE JSC_DECLARE_HOST_FUNCTION(objectProtoFuncToString);
JSString* objectPrototypeToString(JSGlobalObject*, JSValue thisValue);
bool objectPrototypeHasOwnProperty(JSGlobalObject*, JSObject* base, const Identifier& property);

} // namespace JSC
