/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "config.h"
#include "JSRawJSONObject.h"

#include "JSCInlines.h"
#include "PropertyNameArray.h"
#include "TypeError.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSRawJSONObject);

const ClassInfo JSRawJSONObject::s_info = { "Object"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSRawJSONObject) };

JSRawJSONObject::JSRawJSONObject(VM& vm, Structure* structure, Butterfly* butterfly)
    : Base(vm, structure, butterfly)
{
}

void JSRawJSONObject::finishCreation(VM& vm, JSString* string)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    putDirectOffset(vm, rawJSONObjectRawJSONPropertyOffset, string);
}

Structure* JSRawJSONObject::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    auto* structure = Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    structure->addPropertyWithoutTransition(
        vm, vm.propertyNames->rawJSON, PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete,
        [&] (const GCSafeConcurrentJSLocker&, PropertyOffset offset, PropertyOffset newMaxOffset) {
            RELEASE_ASSERT(offset == rawJSONObjectRawJSONPropertyOffset);
            structure->setMaxOffset(vm, newMaxOffset);
        });
    structure->setDidPreventExtensions(true);
    return structure;
}

JSString* JSRawJSONObject::rawJSON(VM& vm)
{
    if (LIKELY(!structure()->didTransition()))
        return jsCast<JSString*>(getDirect(rawJSONObjectRawJSONPropertyOffset));
    return jsCast<JSString*>(getDirect(vm, vm.propertyNames->rawJSON));
}

} // namespace JSC
