/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "JSDOMIterator.h"

#include <JavaScriptCore/ArrayPrototype.h>
#include <JavaScriptCore/BuiltinNames.h>

namespace WebCore {

void addValueIterableMethods(JSC::JSGlobalObject& globalObject, JSC::JSObject& prototype)
{
    JSC::ArrayPrototype* arrayPrototype = globalObject.arrayPrototype();
    ASSERT(arrayPrototype);

    JSC::JSGlobalObject* lexicalGlobalObject = &globalObject;
    ASSERT(lexicalGlobalObject);
    JSC::VM& vm = lexicalGlobalObject->vm();

    auto copyProperty = [&] (const JSC::Identifier& arrayIdentifier, const JSC::Identifier& otherIdentifier, unsigned attributes = 0) {
        JSC::JSValue value = arrayPrototype->getDirect(vm, arrayIdentifier);
        ASSERT(value);
        prototype.putDirect(vm, otherIdentifier, value, attributes);
    };

    copyProperty(vm.propertyNames->builtinNames().entriesPrivateName(), vm.propertyNames->builtinNames().entriesPublicName());
    copyProperty(vm.propertyNames->builtinNames().forEachPrivateName(), vm.propertyNames->builtinNames().forEachPublicName());
    copyProperty(vm.propertyNames->builtinNames().keysPrivateName(), vm.propertyNames->builtinNames().keysPublicName());
    copyProperty(vm.propertyNames->builtinNames().valuesPrivateName(), vm.propertyNames->builtinNames().valuesPublicName());
}

}
