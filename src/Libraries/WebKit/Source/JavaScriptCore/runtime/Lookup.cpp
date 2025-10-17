/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include "Lookup.h"

#include "GetterSetter.h"
#include "StructureInlines.h"
#include <wtf/text/MakeString.h>

namespace JSC {

void reifyStaticAccessor(VM& vm, const HashTableValue& value, JSObject& thisObject, PropertyName propertyName)
{
    JSGlobalObject* globalObject = thisObject.globalObject();
    JSObject* getter = nullptr;
    if (value.hasGetter()) {
        if (value.attributes() & PropertyAttribute::Builtin)
            getter = JSFunction::create(vm, globalObject, value.builtinAccessorGetterGenerator()(vm), globalObject);
        else {
            String getterName = tryMakeString("get "_s, String(*propertyName.publicName()));
            if (!getterName)
                return;
            getter = JSFunction::create(vm, globalObject, 0, getterName, value.accessorGetter(), ImplementationVisibility::Public);
        }
    }
    GetterSetter* accessor = GetterSetter::create(vm, globalObject, getter, nullptr);
    thisObject.putDirectNonIndexAccessor(vm, propertyName, accessor, attributesForStructure(value.attributes()));
}

bool setUpStaticFunctionSlot(VM& vm, const ClassInfo* classInfo, const HashTableValue* entry, JSObject* thisObject, PropertyName propertyName, PropertySlot& slot)
{
    ASSERT(thisObject->globalObject());
    ASSERT(entry->attributes() & PropertyAttribute::BuiltinOrFunctionOrAccessorOrLazyProperty);
    unsigned attributes;
    bool isAccessor = entry->attributes() & PropertyAttribute::Accessor;
    PropertyOffset offset = thisObject->getDirectOffset(vm, propertyName, attributes);

    if (!isValidOffset(offset)) {
        // If a property is ever deleted from an object with a static table, then we reify
        // all static functions at that time - after this we shouldn't be re-adding anything.
        if (thisObject->staticPropertiesReified())
            return false;

        reifyStaticProperty(vm, classInfo, propertyName, *entry, *thisObject);

        offset = thisObject->getDirectOffset(vm, propertyName, attributes);
        if (!isValidOffset(offset)) {
            dataLog("Static hashtable initialiation for ", propertyName, " did not produce a property.\n");
            RELEASE_ASSERT_NOT_REACHED();
        }
    }

    if (isAccessor)
        slot.setCacheableGetterSlot(thisObject, attributes, jsCast<GetterSetter*>(thisObject->getDirect(offset)), offset);
    else
        slot.setValue(thisObject, attributes, thisObject->getDirect(offset), offset);
    return true;
}

} // namespace JSC
