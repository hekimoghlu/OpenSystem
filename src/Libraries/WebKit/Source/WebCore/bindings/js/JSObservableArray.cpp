/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#include "JSObservableArray.h"

#include "JSDOMBinding.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/ArrayPrototype.h>
#include <JavaScriptCore/Error.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>
#include <JavaScriptCore/PropertyNameArray.h>

using namespace WebCore;

namespace JSC {

// https://webidl.spec.whatwg.org/#observable-array-exotic-object-set-the-length
static bool observableArraySetLength(JSObservableArray* object, JSGlobalObject* lexicalGlobalObject, ThrowScope& scope, JSValue value)
{
    auto length = value.toUInt32(lexicalGlobalObject);
    RETURN_IF_EXCEPTION(scope, false);
    auto lengthNumber = value.toNumber(lexicalGlobalObject);
    RETURN_IF_EXCEPTION(scope, false);
    if (length != lengthNumber) {
        throwRangeError(lexicalGlobalObject, scope, "Invalid length"_s);
        return false;
    }
    auto& concreteArray = object->getConcreteArray();
    if (length > concreteArray.length())
        return false;
    concreteArray.shrinkTo(length);
    return true;
}

const ClassInfo JSObservableArray::s_info = { "JSObservableArray"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSObservableArray) };

static JSC_DECLARE_CUSTOM_GETTER(arrayLengthGetter);

JSObservableArray::JSObservableArray(VM& vm, Structure* structure)
    : JSArray(vm, structure, nullptr)
{
}

void JSObservableArray::finishCreation(VM& vm, Ref<ObservableArray>&& array)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    m_array = WTFMove(array);
}

JSObservableArray::~JSObservableArray() = default;

void JSObservableArray::destroy(JSCell* cell)
{
    static_cast<JSObservableArray*>(cell)->JSObservableArray::~JSObservableArray();
}

JSC_DEFINE_CUSTOM_GETTER(arrayLengthGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObservableArray* thisObject = jsDynamicCast<JSObservableArray*>(JSValue::decode(thisValue));
    if (!thisObject)
        return throwVMTypeError(lexicalGlobalObject, scope);
    return JSValue::encode(jsNumber(thisObject->length()));
}

void JSObservableArray::getOwnPropertyNames(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    VM& vm = lexicalGlobalObject->vm();
    JSObservableArray* thisObject = jsCast<JSObservableArray*>(object);
    unsigned length = thisObject->length();
    for (unsigned i = 0; i < length; ++i)
        propertyNames.add(Identifier::from(vm, i));

    if (mode == DontEnumPropertiesMode::Include)
        propertyNames.add(vm.propertyNames->length);

    thisObject->getOwnNonIndexPropertyNames(lexicalGlobalObject, propertyNames, mode);
}

bool JSObservableArray::getOwnPropertySlot(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    VM& vm = lexicalGlobalObject->vm();
    JSObservableArray* thisObject = jsCast<JSObservableArray*>(object);
    if (propertyName == vm.propertyNames->length) {
        slot.setCacheableCustom(thisObject, PropertyAttribute::DontDelete | PropertyAttribute::DontEnum, arrayLengthGetter);
        return true;
    }

    std::optional<uint32_t> index = parseIndex(propertyName);
    if (index && index.value() < thisObject->length()) {
        slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontDelete),
            thisObject->getConcreteArray().valueAt(lexicalGlobalObject, index.value()));
        return true;
    }

    return JSObject::getOwnPropertySlot(thisObject, lexicalGlobalObject, propertyName, slot);
}

bool JSObservableArray::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* lexicalGlobalObject, unsigned index, PropertySlot& slot)
{
    JSObservableArray* thisObject = jsCast<JSObservableArray*>(object);
    if (index < thisObject->length()) {
        slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontDelete),
            thisObject->getConcreteArray().valueAt(lexicalGlobalObject, index));
        return true;
    }

    return JSObject::getOwnPropertySlotByIndex(thisObject, lexicalGlobalObject, index, slot);
}

bool JSObservableArray::put(JSCell* cell, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* thisObject = jsCast<JSObservableArray*>(cell);
    if (propertyName == vm.propertyNames->length)
        return observableArraySetLength(thisObject, lexicalGlobalObject, scope, value);

    if (auto index = parseIndex(propertyName))
        return putByIndex(cell, lexicalGlobalObject, *index, value, slot.isStrictMode());

    RELEASE_AND_RETURN(scope, JSObject::put(thisObject, lexicalGlobalObject, propertyName, value, slot));
}

// https://webidl.spec.whatwg.org/#observable-array-exotic-object-set-the-indexed-value
bool JSObservableArray::putByIndex(JSCell* cell, JSGlobalObject* lexicalGlobalObject, unsigned index, JSValue value, bool)
{
    auto* thisObject = jsCast<JSObservableArray*>(cell);
    auto& concreteArray = thisObject->getConcreteArray();
    if (index > concreteArray.length())
        return false;
    return concreteArray.setValueAt(lexicalGlobalObject, index, value);
}

// https://webidl.spec.whatwg.org/#es-observable-array-deleteProperty
bool JSObservableArray::deleteProperty(JSCell* cell, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (propertyName == vm.propertyNames->length)
        return false;

    if (auto index = parseIndex(propertyName))
        return deletePropertyByIndex(cell, lexicalGlobalObject, *index);

    auto* thisObject = jsCast<JSObservableArray*>(cell);
    RELEASE_AND_RETURN(scope, JSObject::deleteProperty(thisObject, lexicalGlobalObject, propertyName, slot));
}

// https://webidl.spec.whatwg.org/#es-observable-array-deleteProperty
bool JSObservableArray::deletePropertyByIndex(JSCell* cell, JSGlobalObject*, unsigned index)
{
    auto* thisObject = jsCast<JSObservableArray*>(cell);
    auto& concreteArray = thisObject->getConcreteArray();
    if (!concreteArray.length() || index != concreteArray.length() - 1)
        return false;
    concreteArray.removeLast();
    return true;
}

// https://webidl.spec.whatwg.org/#es-observable-array-defineProperty
bool JSObservableArray::defineOwnProperty(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, const PropertyDescriptor& descriptor, bool throwException)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObservableArray* thisObject = jsCast<JSObservableArray*>(object);
    if (propertyName == vm.propertyNames->length) {
        if (descriptor.isAccessorDescriptor())
            return typeError(globalObject, scope, throwException, "Not allowed to change access mechanism for 'length' property"_s);
        if (descriptor.configurablePresent() && descriptor.configurable())
            return typeError(globalObject, scope, throwException, "'length' property must be not configurable"_s);
        if (descriptor.enumerablePresent() && descriptor.enumerable())
            return typeError(globalObject, scope, throwException, "'length' property must be not enumerable"_s);
        if (descriptor.writablePresent() && !descriptor.writable())
            return typeError(globalObject, scope, throwException, "'length' property must be writable"_s);
        if (descriptor.value())
            return observableArraySetLength(thisObject, globalObject, scope, descriptor.value());
        return true;
    }
    if (std::optional<uint32_t> index = parseIndex(propertyName)) {
        if (descriptor.isAccessorDescriptor())
            return typeError(globalObject, scope, throwException, "Not allowed to change access mechanism for an indexed property"_s);
        if (descriptor.configurablePresent() && !descriptor.configurable())
            return typeError(globalObject, scope, throwException, "Indexed property must be configurable"_s);
        if (descriptor.enumerablePresent() && !descriptor.enumerable())
            return typeError(globalObject, scope, throwException, "Indexed property must be enumerable"_s);
        if (descriptor.writablePresent() && !descriptor.writable())
            return typeError(globalObject, scope, throwException, "Indexed property must be writable"_s);
        if (descriptor.value())
            return putByIndex(object, globalObject, *index, descriptor.value(), throwException);
        return true;
    }
    RELEASE_AND_RETURN(scope, Base::defineOwnProperty(object, globalObject, propertyName, descriptor, throwException));
}

JSC::GCClient::IsoSubspace* JSObservableArray::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->observableArraySpace();
}

} // namespace JSC
