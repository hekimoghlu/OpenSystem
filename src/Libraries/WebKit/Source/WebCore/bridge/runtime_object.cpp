/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "runtime_object.h"

#include "JSDOMBinding.h"
#include "WebCoreJSClientData.h"
#include "runtime_method.h"
#include <JavaScriptCore/Error.h>

using namespace WebCore;

namespace JSC {
namespace Bindings {

WEBCORE_EXPORT const ClassInfo RuntimeObject::s_info = { "RuntimeObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(RuntimeObject) };

static JSC_DECLARE_HOST_FUNCTION(convertRuntimeObjectToPrimitive);
static JSC_DECLARE_HOST_FUNCTION(callRuntimeObject);
static JSC_DECLARE_HOST_FUNCTION(callRuntimeConstructor);

static JSC_DECLARE_CUSTOM_GETTER(fallbackObjectGetter);
static JSC_DECLARE_CUSTOM_GETTER(fieldGetter);
static JSC_DECLARE_CUSTOM_GETTER(methodGetter);

RuntimeObject::RuntimeObject(VM& vm, Structure* structure, RefPtr<Instance>&& instance)
    : Base(vm, structure)
    , m_instance(WTFMove(instance))
{
}

void RuntimeObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    putDirect(vm, vm.propertyNames->toPrimitiveSymbol,
        JSFunction::create(vm, globalObject(), 1, "[Symbol.toPrimitive]"_s, convertRuntimeObjectToPrimitive, ImplementationVisibility::Public),
        enumToUnderlyingType(PropertyAttribute::DontEnum));
}

void RuntimeObject::destroy(JSCell* cell)
{
    static_cast<RuntimeObject*>(cell)->RuntimeObject::~RuntimeObject();
}

void RuntimeObject::invalidate()
{
    ASSERT(m_instance);
    if (m_instance)
        m_instance->willInvalidateRuntimeObject();
    m_instance = nullptr;
}

JSC_DEFINE_CUSTOM_GETTER(fallbackObjectGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeObject* thisObj = jsCast<RuntimeObject*>(JSValue::decode(thisValue));
    RefPtr<Instance> instance = thisObj->getInternalInstance();

    if (!instance)
        return JSValue::encode(throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope));
    
    instance->begin();

    Class *aClass = instance->getClass();
    JSValue result = aClass->fallbackObject(lexicalGlobalObject, instance.get(), propertyName);

    instance->end();
            
    return JSValue::encode(result);
}

JSC_DEFINE_CUSTOM_GETTER(fieldGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeObject* thisObj = jsCast<RuntimeObject*>(JSValue::decode(thisValue));
    RefPtr<Instance> instance = thisObj->getInternalInstance();

    if (!instance)
        return JSValue::encode(throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope));
    
    instance->begin();

    Class *aClass = instance->getClass();
    Field* aField = aClass->fieldNamed(propertyName, instance.get());
    JSValue result = aField->valueFromInstance(lexicalGlobalObject, instance.get());
    
    instance->end();
            
    return JSValue::encode(result);
}

JSC_DEFINE_CUSTOM_GETTER(methodGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeObject* thisObj = jsCast<RuntimeObject*>(JSValue::decode(thisValue));
    RefPtr<Instance> instance = thisObj->getInternalInstance();

    if (!instance)
        return JSValue::encode(throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope));
    
    instance->begin();

    JSValue method = instance->getMethod(lexicalGlobalObject, propertyName);

    instance->end();
            
    return JSValue::encode(method);
}

bool RuntimeObject::getOwnPropertySlot(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    Ref vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    RuntimeObject* thisObject = jsCast<RuntimeObject*>(object);
    if (!thisObject->m_instance) {
        throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope);
        return false;
    }

    if (propertyName.uid() == vm->propertyNames->toPrimitiveSymbol.impl())
        return JSObject::getOwnPropertySlot(thisObject, lexicalGlobalObject, propertyName, slot);
    
    RefPtr<Instance> instance = thisObject->m_instance;

    instance->begin();
    
    Class *aClass = instance->getClass();
    
    if (aClass) {
        // See if the instance has a field with the specified name.
        Field *aField = aClass->fieldNamed(propertyName, instance.get());
        if (aField) {
            slot.setCustom(thisObject, enumToUnderlyingType(JSC::PropertyAttribute::DontDelete), fieldGetter);
            instance->end();
            return true;
        } else {
            // Now check if a method with specified name exists, if so return a function object for
            // that method.
            if (aClass->methodNamed(propertyName, instance.get())) {
                slot.setCustom(thisObject, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly, methodGetter);
                
                instance->end();
                return true;
            }
        }

        // Try a fallback object.
        if (!aClass->fallbackObject(lexicalGlobalObject, instance.get(), propertyName).isUndefined()) {
            slot.setCustom(thisObject, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum, fallbackObjectGetter);
            instance->end();
            return true;
        }
    }
        
    instance->end();
    
    return instance->getOwnPropertySlot(thisObject, lexicalGlobalObject, propertyName, slot);
}

bool RuntimeObject::put(JSCell* cell, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeObject* thisObject = jsCast<RuntimeObject*>(cell);
    if (!thisObject->m_instance) {
        throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope);
        return false;
    }
    
    RefPtr<Instance> instance = thisObject->m_instance;
    instance->begin();

    // Set the value of the property.
    bool result = false;
    Field *aField = instance->getClass()->fieldNamed(propertyName, instance.get());
    if (aField)
        result = aField->setValueToInstance(lexicalGlobalObject, instance.get(), value);
    else if (!instance->setValueOfUndefinedField(lexicalGlobalObject, propertyName, value))
        result = instance->put(thisObject, lexicalGlobalObject, propertyName, value, slot);

    instance->end();
    return result;
}

bool RuntimeObject::deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&)
{
    // Can never remove a property of a RuntimeObject.
    return false;
}

JSC_DEFINE_HOST_FUNCTION(convertRuntimeObjectToPrimitive, (JSGlobalObject* lexicalGlobalObject, CallFrame* callFrame))
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    auto* thisObject = jsDynamicCast<RuntimeObject*>(callFrame->thisValue());
    if (!thisObject)
        return throwVMTypeError(lexicalGlobalObject, scope, "RuntimeObject[Symbol.toPrimitive] method called on incompatible |this| value."_s);
    RefPtr instance = thisObject->getInternalInstance();
    if (!instance)
        return JSValue::encode(throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope));
    auto hint = toPreferredPrimitiveType(lexicalGlobalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, { });

    instance->begin();
    JSValue result = instance->defaultValue(lexicalGlobalObject, hint);
    instance->end();
    return JSValue::encode(result);
}

JSC_DEFINE_HOST_FUNCTION(callRuntimeObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ASSERT_UNUSED(globalObject, callFrame->jsCallee()->inherits<RuntimeObject>());
    RefPtr<Instance> instance(static_cast<RuntimeObject*>(callFrame->jsCallee())->getInternalInstance());
    instance->begin();
    JSValue result = instance->invokeDefaultMethod(globalObject, callFrame);
    instance->end();
    return JSValue::encode(result);
}

CallData RuntimeObject::getCallData(JSCell* cell)
{
    CallData callData;

    RuntimeObject* thisObject = jsCast<RuntimeObject*>(cell);
    if (thisObject->m_instance && thisObject->m_instance->supportsInvokeDefaultMethod()) {
        callData.type = CallData::Type::Native;
        callData.native.function = callRuntimeObject;
        callData.native.isBoundFunction = false;
        callData.native.isWasm = false;
    }

    return callData;
}

JSC_DEFINE_HOST_FUNCTION(callRuntimeConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    JSObject* constructor = callFrame->jsCallee();
    ASSERT_UNUSED(globalObject, constructor->inherits<RuntimeObject>());
    RefPtr<Instance> instance(static_cast<RuntimeObject*>(callFrame->jsCallee())->getInternalInstance());
    instance->begin();
    ArgList args(callFrame);
    JSValue result = instance->invokeConstruct(globalObject, callFrame, args);
    instance->end();
    
    ASSERT(result);
    return JSValue::encode(result.isObject() ? jsCast<JSObject*>(result.asCell()) : constructor);
}

CallData RuntimeObject::getConstructData(JSCell* cell)
{
    CallData constructData;

    RuntimeObject* thisObject = jsCast<RuntimeObject*>(cell);
    if (thisObject->m_instance && thisObject->m_instance->supportsConstruct()) {
        constructData.type = CallData::Type::Native;
        constructData.native.function = callRuntimeConstructor;
        constructData.native.isBoundFunction = false;
        constructData.native.isWasm = false;
    }

    return constructData;
}

void RuntimeObject::getOwnPropertyNames(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode)
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());

    RuntimeObject* thisObject = jsCast<RuntimeObject*>(object);
    if (!thisObject->m_instance) {
        throwRuntimeObjectInvalidAccessError(lexicalGlobalObject, scope);
        return;
    }

    RefPtr<Instance> instance = thisObject->m_instance;
    
    instance->begin();
    instance->getPropertyNames(lexicalGlobalObject, propertyNames);
    instance->end();
}

Exception* throwRuntimeObjectInvalidAccessError(JSGlobalObject* lexicalGlobalObject, ThrowScope& scope)
{
    return throwException(lexicalGlobalObject, scope, createReferenceError(lexicalGlobalObject, "Trying to access object from destroyed plug-in."_s));
}

JSC::GCClient::IsoSubspace* RuntimeObject::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->runtimeObjectSpace();
}

}
}
