/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

namespace JSC {

constexpr PropertyOffset rawJSONObjectRawJSONPropertyOffset = firstOutOfLineOffset;

class JSRawJSONObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    using Base::StructureFlags;

    template<typename, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.rawJSONObjectSpace<mode>();
    }

    static JSRawJSONObject* tryCreate(VM& vm, Structure* structure, JSString* string)
    {
        constexpr bool hasIndexingHeader = false;
        Butterfly* butterfly = Butterfly::tryCreate(vm, nullptr, 0, structure->outOfLineCapacity(), hasIndexingHeader, IndexingHeader(), 0);
        if (!butterfly)
            return nullptr;
        JSRawJSONObject* object = new (NotNull, allocateCell<JSRawJSONObject>(vm)) JSRawJSONObject(vm, structure, butterfly);
        object->finishCreation(vm, string);
        return object;
    }

    DECLARE_EXPORT_INFO;

    JSString* rawJSON(VM&);

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

protected:
    JS_EXPORT_PRIVATE void finishCreation(VM&, JSString*);
    JS_EXPORT_PRIVATE JSRawJSONObject(VM&, Structure*, Butterfly*);
};

} // namespace JSC
