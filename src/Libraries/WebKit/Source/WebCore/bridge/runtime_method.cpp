/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include "runtime_method.h"

#include "JSDOMBinding.h"
#include "JSHTMLElement.h"
#include "JSPluginElementFunctions.h"
#include "WebCoreJSClientData.h"
#include "runtime_object.h"
#include <JavaScriptCore/Error.h>
#include <JavaScriptCore/FunctionPrototype.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>

using namespace WebCore;

namespace JSC {

using namespace Bindings;

WEBCORE_EXPORT const ClassInfo RuntimeMethod::s_info = { "RuntimeMethod"_s, &InternalFunction::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(RuntimeMethod) };

static JSC_DECLARE_HOST_FUNCTION(callRuntimeMethod);
static JSC_DECLARE_CUSTOM_GETTER(methodLengthGetter);

RuntimeMethod::RuntimeMethod(VM& vm, Structure* structure, Method* method)
    // Callers will need to pass in the right global object corresponding to this native object "method".
    : InternalFunction(vm, structure, callRuntimeMethod, nullptr)
    , m_method(method)
{
}

void RuntimeMethod::finishCreation(VM& vm, const String& ident)
{
    Base::finishCreation(vm, 0, ident);
    ASSERT(inherits(info()));
}

JSC_DEFINE_CUSTOM_GETTER(methodLengthGetter, (JSGlobalObject* exec, EncodedJSValue thisValue, PropertyName))
{
    auto scope = DECLARE_THROW_SCOPE(exec->vm());

    RuntimeMethod* thisObject = jsDynamicCast<RuntimeMethod*>(JSValue::decode(thisValue));
    if (!thisObject)
        return throwVMTypeError(exec, scope);
    return JSValue::encode(jsNumber(thisObject->method()->numParameters()));
}

bool RuntimeMethod::getOwnPropertySlot(JSObject* object, JSGlobalObject* exec, PropertyName propertyName, PropertySlot &slot)
{
    Ref vm = exec->vm();
    RuntimeMethod* thisObject = jsCast<RuntimeMethod*>(object);
    if (propertyName == vm->propertyNames->length) {
        slot.setCacheableCustom(thisObject, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum, methodLengthGetter);
        return true;
    }
    
    return InternalFunction::getOwnPropertySlot(thisObject, exec, propertyName, slot);
}

GCClient::IsoSubspace* RuntimeMethod::subspaceForImpl(VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->runtimeMethodSpace();
}

JSC_DEFINE_HOST_FUNCTION(callRuntimeMethod, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());

    RuntimeMethod* method = static_cast<RuntimeMethod*>(callFrame->jsCallee());

    if (!method->method())
        return JSValue::encode(jsUndefined());

    RefPtr<Instance> instance;

    JSValue thisValue = callFrame->thisValue();
    if (thisValue.inherits<RuntimeObject>()) {
        RuntimeObject* runtimeObject = static_cast<RuntimeObject*>(asObject(thisValue));
        instance = runtimeObject->getInternalInstance();
        if (!instance) 
            return JSValue::encode(throwRuntimeObjectInvalidAccessError(globalObject, scope));
    } else {
        // Calling a runtime object of a plugin element?
        if (thisValue.inherits<JSHTMLElement>())
            instance = pluginInstance(jsCast<JSHTMLElement*>(asObject(thisValue))->wrapped());
        if (!instance)
            return throwVMTypeError(globalObject, scope);
    }
    ASSERT(instance);

    instance->begin();
    JSValue result = instance->invokeMethod(globalObject, callFrame, method);
    instance->end();
    return JSValue::encode(result);
}

}
