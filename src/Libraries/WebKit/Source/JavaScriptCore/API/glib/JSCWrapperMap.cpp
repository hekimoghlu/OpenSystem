/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include "JSCWrapperMap.h"

#include "APICast.h"
#include "IntegrityInlines.h"
#include "JSAPIWrapperGlobalObject.h"
#include "JSAPIWrapperObject.h"
#include "JSCClassPrivate.h"
#include "JSCContextPrivate.h"
#include "JSCGLibWrapperObject.h"
#include "JSCInlines.h"
#include "JSCValuePrivate.h"
#include "JSCallbackObject.h"
#include "JSContextRef.h"

namespace JSC {

WrapperMap::WrapperMap(JSGlobalContextRef jsContext)
    : m_cachedJSWrappers(makeUnique<JSC::WeakGCMap<gpointer, JSC::JSObject>>(toJS(jsContext)->vm()))
{
}

WrapperMap::~WrapperMap()
{
    for (const auto& jscClass : m_classMap.values())
        jscClassInvalidate(jscClass.get());
}

GRefPtr<JSCValue> WrapperMap::gobjectWrapper(JSCContext* jscContext, JSValueRef jsValue)
{
    auto* jsContext = jscContextGetJSContext(jscContext);
    JSC::JSLockHolder locker(toJS(jsContext));
    ASSERT(toJSGlobalObject(jsContext)->wrapperMap() == this);
    GRefPtr<JSCValue> value = m_cachedGObjectWrappers.get(jsValue);
    if (!value) {
        value = adoptGRef(jscValueCreate(jscContext, jsValue));
        m_cachedGObjectWrappers.set(jsValue, value.get());
    }
    return value;
}

void WrapperMap::unwrap(JSValueRef jsValue)
{
    ASSERT(m_cachedGObjectWrappers.contains(jsValue));
    m_cachedGObjectWrappers.remove(jsValue);
}

void WrapperMap::registerClass(JSCClass* jscClass)
{
    RefPtr jsClass = jscClassGetJSClass(jscClass);
    ASSERT(!m_classMap.contains(jsClass.get()));
    m_classMap.set(jsClass.get(), jscClass);
}

JSCClass* WrapperMap::registeredClass(JSClassRef jsClass) const
{
    return m_classMap.get(jsClass);
}

JSObject* WrapperMap::createJSWrapper(JSGlobalContextRef jsContext, JSClassRef jsClass, JSValueRef prototype, gpointer wrappedObject, GDestroyNotify destroyFunction)
{
    ASSERT(toJSGlobalObject(jsContext)->wrapperMap() == this);
    JSGlobalObject* globalObject = toJS(jsContext);
    Ref vm = globalObject->vm();
    JSLockHolder locker(vm);
    auto* object = JSC::JSCallbackObject<JSC::JSAPIWrapperObject>::create(globalObject, globalObject->glibWrapperObjectStructure(), jsClass, nullptr);
    if (wrappedObject) {
        object->setWrappedObject(new JSC::JSCGLibWrapperObject(wrappedObject, destroyFunction));
        m_cachedJSWrappers->set(wrappedObject, object);
    }
    if (prototype)
        JSObjectSetPrototype(jsContext, toRef(object), prototype);
    else if (auto* jsPrototype = jsClass->prototype(globalObject))
        object->setPrototypeDirect(vm, jsPrototype);
    return object;
}

JSGlobalContextRef WrapperMap::createContextWithJSWrapper(JSContextGroupRef jsGroup, JSClassRef jsClass, JSValueRef prototype, gpointer wrappedObject, GDestroyNotify destroyFunction)
{
    Ref<VM> vm(*toJS(jsGroup));
    JSLockHolder locker(vm.ptr());
    auto* globalObject = JSCallbackObject<JSAPIWrapperGlobalObject>::create(vm.get(), jsClass, JSCallbackObject<JSAPIWrapperGlobalObject>::createStructure(vm.get(), nullptr, jsNull()));
    if (wrappedObject) {
        globalObject->setWrappedObject(new JSC::JSCGLibWrapperObject(wrappedObject, destroyFunction));
        m_cachedJSWrappers->set(wrappedObject, globalObject);
    }
    if (prototype)
        globalObject->resetPrototype(vm.get(), toJS(globalObject, prototype));
    else if (auto jsPrototype = jsClass->prototype(globalObject))
        globalObject->resetPrototype(vm.get(), jsPrototype);
    else
        globalObject->resetPrototype(vm.get(), jsNull());

    return JSGlobalContextRetain(toGlobalRef(globalObject));
}

JSObject* WrapperMap::jsWrapper(gpointer wrappedObject) const
{
    if (!wrappedObject)
        return nullptr;
    return m_cachedJSWrappers->get(wrappedObject);
}

gpointer WrapperMap::wrappedObject(JSGlobalContextRef jsContext, JSObjectRef jsObject) const
{
    ASSERT(toJSGlobalObject(jsContext)->wrapperMap() == this);
    JSLockHolder locker(toJS(jsContext));
    auto* object = toJS(jsObject);
    if (object->inherits<JSC::JSCallbackObject<JSC::JSAPIWrapperObject>>()) {
        if (auto* wrapper = JSC::jsCast<JSC::JSAPIWrapperObject*>(object)->wrappedObject())
            return static_cast<JSC::JSCGLibWrapperObject*>(wrapper)->object();
    }
    if (object->inherits<JSC::JSCallbackObject<JSC::JSAPIWrapperGlobalObject>>()) {
        if (auto* wrapper = JSC::jsCast<JSC::JSAPIWrapperGlobalObject*>(object)->wrappedObject())
            return wrapper->object();
    }
    return nullptr;
}

} // namespace JSC
