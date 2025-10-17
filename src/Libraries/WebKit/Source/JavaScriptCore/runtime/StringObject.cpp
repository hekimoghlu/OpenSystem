/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "StringObject.h"

#include "JSCInlines.h"
#include "PropertyNameArray.h"
#include "TypeError.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(StringObject);

const ClassInfo StringObject::s_info = { "String"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(StringObject) };

StringObject::StringObject(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void StringObject::finishCreation(VM& vm, JSString* string)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    setInternalValue(vm, string);
    ASSERT_WITH_MESSAGE(type() == StringObjectType || type() == DerivedStringObjectType, "Instance inheriting StringObject should have DerivedStringObjectType");
}

bool StringObject::getOwnPropertySlot(JSObject* cell, JSGlobalObject* globalObject, PropertyName propertyName, PropertySlot& slot)
{
    StringObject* thisObject = jsCast<StringObject*>(cell);
    if (thisObject->internalValue()->getStringPropertySlot(globalObject, propertyName, slot))
        return true;
    return JSObject::getOwnPropertySlot(thisObject, globalObject, propertyName, slot);
}
    
bool StringObject::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* globalObject, unsigned propertyName, PropertySlot& slot)
{
    StringObject* thisObject = jsCast<StringObject*>(object);
    if (thisObject->internalValue()->getStringPropertySlot(globalObject, propertyName, slot))
        return true;    
    VM& vm = globalObject->vm();
    return JSObject::getOwnPropertySlot(thisObject, globalObject, Identifier::from(vm, propertyName), slot);
}

bool StringObject::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    StringObject* thisObject = jsCast<StringObject*>(cell);

    if (propertyName == vm.propertyNames->length)
        return typeError(globalObject, scope, slot.isStrictMode(), ReadonlyPropertyWriteError);
    if (UNLIKELY(slot.thisValue() != thisObject))
        RELEASE_AND_RETURN(scope, JSObject::put(cell, globalObject, propertyName, value, slot));
    if (std::optional<uint32_t> index = parseIndex(propertyName)) 
        RELEASE_AND_RETURN(scope, putByIndex(cell, globalObject, index.value(), value, slot.isStrictMode()));
    RELEASE_AND_RETURN(scope, JSObject::put(cell, globalObject, propertyName, value, slot));
}

bool StringObject::putByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned propertyName, JSValue value, bool shouldThrow)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    StringObject* thisObject = jsCast<StringObject*>(cell);
    if (thisObject->internalValue()->canGetIndex(propertyName))
        return typeError(globalObject, scope, shouldThrow, ReadonlyPropertyWriteError);
    RELEASE_AND_RETURN(scope, JSObject::putByIndex(cell, globalObject, propertyName, value, shouldThrow));
}

static bool isStringOwnProperty(JSGlobalObject* globalObject, StringObject* object, PropertyName propertyName)
{
    VM& vm = globalObject->vm();
    if (propertyName == vm.propertyNames->length)
        return true;
    if (std::optional<uint32_t> index = parseIndex(propertyName)) {
        if (object->internalValue()->canGetIndex(index.value()))
            return true;
    }
    return false;
}

bool StringObject::defineOwnProperty(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, const PropertyDescriptor& descriptor, bool throwException)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    StringObject* thisObject = jsCast<StringObject*>(object);

    if (isStringOwnProperty(globalObject, thisObject, propertyName)) {
        // The current PropertyDescriptor is always
        // PropertyDescriptor{[[Value]]: value, [[Writable]]: false, [[Enumerable]]: true, [[Configurable]]: false}.
        // This ensures that any property descriptor cannot change the existing one.
        // Here, simply return the result of validateAndApplyPropertyDescriptor.
        // https://tc39.github.io/ecma262/#sec-string-exotic-objects-getownproperty-p
        PropertyDescriptor current;
        bool isCurrentDefined = thisObject->getOwnPropertyDescriptor(globalObject, propertyName, current);
        EXCEPTION_ASSERT(!scope.exception() == isCurrentDefined);
        RETURN_IF_EXCEPTION(scope, false);
        bool isExtensible = thisObject->isExtensible(globalObject);
        RETURN_IF_EXCEPTION(scope, false);
        RELEASE_AND_RETURN(scope, validateAndApplyPropertyDescriptor(globalObject, nullptr, propertyName, isExtensible, descriptor, isCurrentDefined, current, throwException));
    }

    RELEASE_AND_RETURN(scope, Base::defineOwnProperty(object, globalObject, propertyName, descriptor, throwException));
}

bool StringObject::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    VM& vm = globalObject->vm();
    StringObject* thisObject = jsCast<StringObject*>(cell);
    if (propertyName == vm.propertyNames->length)
        return false;
    std::optional<uint32_t> index = parseIndex(propertyName);
    if (index && thisObject->internalValue()->canGetIndex(index.value()))
        return false;
    return JSObject::deleteProperty(thisObject, globalObject, propertyName, slot);
}

bool StringObject::deletePropertyByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned i)
{
    StringObject* thisObject = jsCast<StringObject*>(cell);
    if (thisObject->internalValue()->canGetIndex(i))
        return false;
    return JSObject::deletePropertyByIndex(thisObject, globalObject, i);
}

void StringObject::getOwnPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    VM& vm = globalObject->vm();
    StringObject* thisObject = jsCast<StringObject*>(object);
    if (propertyNames.includeStringProperties()) {
        int size = thisObject->internalValue()->length();
        for (int i = 0; i < size; ++i)
            propertyNames.add(Identifier::from(vm, i));
        thisObject->getOwnIndexedPropertyNames(globalObject, propertyNames, mode);
    }
    if (mode == DontEnumPropertiesMode::Include)
        propertyNames.add(vm.propertyNames->length);
    thisObject->getOwnNonIndexPropertyNames(globalObject, propertyNames, mode);
}

StringObject* constructString(VM& vm, JSGlobalObject* globalObject, JSValue string)
{
    StringObject* object = StringObject::create(vm, globalObject->stringObjectStructure());
    object->setInternalValue(vm, string);
    return object;
}

} // namespace JSC
