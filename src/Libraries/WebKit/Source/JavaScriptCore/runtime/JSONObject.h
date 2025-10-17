/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

class JSONObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSONObject, Base);
        return &vm.plainObjectSpace();
    }

    static JSONObject* create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
    {
        JSONObject* object = new (NotNull, allocateCell<JSONObject>(vm)) JSONObject(vm, structure);
        object->finishCreation(vm, globalObject);
        return object;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    JSONObject(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JS_EXPORT_PRIVATE JSValue JSONParse(JSGlobalObject*, StringView);
JSValue JSONParseWithException(JSGlobalObject*, StringView);
JS_EXPORT_PRIVATE String JSONStringify(JSGlobalObject*, JSValue, JSValue space);
JS_EXPORT_PRIVATE String JSONStringify(JSGlobalObject*, JSValue, unsigned indent);
    
} // namespace JSC
