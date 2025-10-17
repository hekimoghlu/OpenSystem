/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "runtime_array.h"

#include "JSDOMBinding.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/ArrayPrototype.h>
#include <JavaScriptCore/Error.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>
#include <JavaScriptCore/PropertyNameArray.h>

using namespace WebCore;

namespace JSC {

const ClassInfo RuntimeArray::s_info = { "RuntimeArray"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(RuntimeArray) };

static JSC_DECLARE_CUSTOM_GETTER(arrayLengthGetter);

RuntimeArray::RuntimeArray(VM& vm, Structure* structure)
    : JSArray(vm, structure, nullptr)
    , m_array(nullptr)
{
}

void RuntimeArray::finishCreation(VM& vm, Bindings::Array* array)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    m_array = array;
}

RuntimeArray::~RuntimeArray()
{
    delete getConcreteArray();
}

void RuntimeArray::destroy(JSCell* cell)
{
    static_cast<RuntimeArray*>(cell)->RuntimeArray::~RuntimeArray();
}

JSC_DEFINE_CUSTOM_GETTER(arrayLengthGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName))
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeArray* thisObject = jsDynamicCast<RuntimeArray*>(JSValue::decode(thisValue));
    if (!thisObject)
        return throwVMTypeError(lexicalGlobalObject, scope);
    return JSValue::encode(jsNumber(thisObject->getLength()));
}

void RuntimeArray::getOwnPropertyNames(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    Ref vm = lexicalGlobalObject->vm();
    RuntimeArray* thisObject = jsCast<RuntimeArray*>(object);
    unsigned length = thisObject->getLength();
    for (unsigned i = 0; i < length; ++i)
        propertyNames.add(Identifier::from(vm, i));

    if (mode == DontEnumPropertiesMode::Include)
        propertyNames.add(vm->propertyNames->length);

    thisObject->getOwnNonIndexPropertyNames(lexicalGlobalObject, propertyNames, mode);
}

bool RuntimeArray::getOwnPropertySlot(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    Ref vm = lexicalGlobalObject->vm();
    RuntimeArray* thisObject = jsCast<RuntimeArray*>(object);
    if (propertyName == vm->propertyNames->length) {
        slot.setCacheableCustom(thisObject, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum, arrayLengthGetter);
        return true;
    }
    
    std::optional<uint32_t> index = parseIndex(propertyName);
    if (index && index.value() < thisObject->getLength()) {
        slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontDelete),
            thisObject->getConcreteArray()->valueAt(lexicalGlobalObject, index.value()));
        return true;
    }
    
    return JSObject::getOwnPropertySlot(thisObject, lexicalGlobalObject, propertyName, slot);
}

bool RuntimeArray::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* lexicalGlobalObject, unsigned index, PropertySlot& slot)
{
    RuntimeArray* thisObject = jsCast<RuntimeArray*>(object);
    if (index < thisObject->getLength()) {
        slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontDelete),
            thisObject->getConcreteArray()->valueAt(lexicalGlobalObject, index));
        return true;
    }
    
    return JSObject::getOwnPropertySlotByIndex(thisObject, lexicalGlobalObject, index, slot);
}

bool RuntimeArray::put(JSCell* cell, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    Ref vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    RuntimeArray* thisObject = jsCast<RuntimeArray*>(cell);
    if (propertyName == vm->propertyNames->length) {
        throwException(lexicalGlobalObject, scope, createRangeError(lexicalGlobalObject, "Range error"_s));
        return false;
    }
    
    if (std::optional<uint32_t> index = parseIndex(propertyName))
        return thisObject->getConcreteArray()->setValueAt(lexicalGlobalObject, index.value(), value);

    RELEASE_AND_RETURN(scope, JSObject::put(thisObject, lexicalGlobalObject, propertyName, value, slot));
}

bool RuntimeArray::putByIndex(JSCell* cell, JSGlobalObject* lexicalGlobalObject, unsigned index, JSValue value, bool)
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeArray* thisObject = jsCast<RuntimeArray*>(cell);
    if (index >= thisObject->getLength()) {
        throwException(lexicalGlobalObject, scope, createRangeError(lexicalGlobalObject, "Range error"_s));
        return false;
    }
    
    return thisObject->getConcreteArray()->setValueAt(lexicalGlobalObject, index, value);
}

bool RuntimeArray::deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&)
{
    return false;
}

bool RuntimeArray::deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned)
{
    return false;
}

JSC::GCClient::IsoSubspace* RuntimeArray::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->runtimeArraySpace();
}

}
