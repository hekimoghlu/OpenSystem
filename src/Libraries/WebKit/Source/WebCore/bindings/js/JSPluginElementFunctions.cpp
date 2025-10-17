/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#include "JSPluginElementFunctions.h"

#include "BridgeJSC.h"
#include "HTMLNames.h"
#include "HTMLPlugInElement.h"
#include "JSHTMLElement.h"

namespace WebCore {
using namespace JSC;

using namespace Bindings;
using namespace HTMLNames;

// JavaScript access to plug-in-exported properties for JSHTMLEmbedElement and JSHTMLObjectElement.

static JSC_DECLARE_HOST_FUNCTION(callPlugin);
static JSC_DECLARE_CUSTOM_GETTER(pluginElementPropertyGetter);

Instance* pluginInstance(HTMLElement& element)
{
    // The plugin element holds an owning reference, so we don't have to.
    auto* pluginElement = dynamicDowncast<HTMLPlugInElement>(element);
    if (!pluginElement)
        return nullptr;
    auto* instance = pluginElement->bindingsInstance();
    if (!instance || !instance->rootObject())
        return nullptr;
    return instance;
}

JSObject* pluginScriptObject(JSGlobalObject* lexicalGlobalObject, JSHTMLElement* jsHTMLElement)
{
    auto* element = dynamicDowncast<HTMLPlugInElement>(jsHTMLElement->wrapped());
    if (!element)
        return nullptr;

    // Choke point for script/plugin interaction; notify DOMTimer of the event.
    DOMTimer::scriptDidInteractWithPlugin();

    // The plugin element holds an owning reference, so we don't have to.
    auto* instance = element->bindingsInstance();
    if (!instance || !instance->rootObject())
        return nullptr;

    return instance->createRuntimeObject(lexicalGlobalObject);
}

JSC_DEFINE_CUSTOM_GETTER(pluginElementPropertyGetter, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSHTMLElement* thisObject = jsDynamicCast<JSHTMLElement*>(JSValue::decode(thisValue));
    if (!thisObject)
        return throwVMTypeError(lexicalGlobalObject, scope);
    JSObject* scriptObject = pluginScriptObject(lexicalGlobalObject, thisObject);
    if (!scriptObject)
        return JSValue::encode(jsUndefined());
    
    return JSValue::encode(scriptObject->get(lexicalGlobalObject, propertyName));
}

bool pluginElementCustomGetOwnPropertySlot(JSHTMLElement* element, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    VM& vm = lexicalGlobalObject->vm();
    slot.setIsTaintedByOpaqueObject();

    if (propertyName.uid() == vm.propertyNames->toPrimitiveSymbol.impl())
        return false;

    if (!element->globalObject()->world().isNormal()) {
        JSValue proto = element->getPrototypeDirect();
        if (proto.isObject() && JSC::jsCast<JSC::JSObject*>(asObject(proto))->hasProperty(lexicalGlobalObject, propertyName))
            return false;
    }

    if (slot.isVMInquiry()) {
        slot.setValue(element, enumToUnderlyingType(JSC::PropertyAttribute::None), jsUndefined());
        return false; // Can't execute stuff below because they can call back into JS.
    }

    JSObject* scriptObject = pluginScriptObject(lexicalGlobalObject, element);
    if (!scriptObject)
        return false;

    if (!scriptObject->hasProperty(lexicalGlobalObject, propertyName))
        return false;

    slot.setCustom(element, JSC::PropertyAttribute::DontDelete | JSC::PropertyAttribute::DontEnum, pluginElementPropertyGetter);
    return true;
}

bool pluginElementCustomPut(JSHTMLElement* element, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot, bool& putResult)
{
    JSObject* scriptObject = pluginScriptObject(lexicalGlobalObject, element);
    if (!scriptObject)
        return false;
    if (!scriptObject->hasProperty(lexicalGlobalObject, propertyName))
        return false;
    putResult = scriptObject->methodTable()->put(scriptObject, lexicalGlobalObject, propertyName, value, slot);
    return true;
}

JSC_DEFINE_HOST_FUNCTION(callPlugin, (JSGlobalObject* lexicalGlobalObject, CallFrame* callFrame))
{
    JSHTMLElement* element = jsCast<JSHTMLElement*>(callFrame->jsCallee());

    // Get the plug-in script object.
    JSObject* scriptObject = pluginScriptObject(lexicalGlobalObject, element);
    ASSERT(scriptObject);

    size_t argumentCount = callFrame->argumentCount();
    MarkedArgumentBuffer argumentList;
    argumentList.ensureCapacity(argumentCount);
    for (size_t i = 0; i < argumentCount; i++)
        argumentList.append(callFrame->argument(i));
    ASSERT(!argumentList.hasOverflowed());

    auto callData = JSC::getCallData(scriptObject);
    ASSERT(callData.type == CallData::Type::Native);

    // Call the object.
    JSValue result = call(lexicalGlobalObject, scriptObject, callData, callFrame->thisValue(), argumentList);
    return JSValue::encode(result);
}

CallData pluginElementCustomGetCallData(JSHTMLElement* element)
{
    CallData callData;

    Instance* instance = pluginInstance(element->wrapped());
    if (instance && instance->supportsInvokeDefaultMethod()) {
        callData.type = CallData::Type::Native;
        callData.native.function = callPlugin;
        callData.native.isBoundFunction = false;
        callData.native.isWasm = false;
    }

    return callData;
}

} // namespace WebCore
