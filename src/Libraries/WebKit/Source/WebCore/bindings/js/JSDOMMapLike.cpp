/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "JSDOMMapLike.h"

#include "WebCoreJSBuiltinInternals.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/JSMap.h>
#include <JavaScriptCore/VMTrapsInlines.h>

namespace WebCore {

std::pair<bool, std::reference_wrapper<JSC::JSObject>> getBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& mapLike)
{
    auto& vm = lexicalGlobalObject.vm();
    auto backingMap = mapLike.getDirect(vm, builtinNames(vm).backingMapPrivateName());
    if (backingMap)
        return { false, *JSC::asObject(backingMap) };

    backingMap = JSC::JSMap::create(vm, lexicalGlobalObject.mapStructure());
    mapLike.putDirect(vm, builtinNames(vm).backingMapPrivateName(), backingMap, enumToUnderlyingType(JSC::PropertyAttribute::DontEnum));
    return { true, *JSC::asObject(backingMap) };
}

void clearBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingMap)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.mapPrototype()->getDirect(vm, vm.propertyNames->builtinNames().clearPrivateName());
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);

    JSC::MarkedArgumentBuffer arguments;
    JSC::call(&lexicalGlobalObject, function, callData, &backingMap, arguments);
}

void setToBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingMap, JSC::JSValue key, JSC::JSValue value)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.mapPrototype()->getDirect(vm, vm.propertyNames->builtinNames().setPrivateName());
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(key);
    arguments.append(value);
    ASSERT(!arguments.hasOverflowed());
    JSC::call(&lexicalGlobalObject, function, callData, &backingMap, arguments);
}

JSC::JSValue forwardAttributeGetterToBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingMap, const JSC::Identifier& attributeName)
{
    return backingMap.get(&lexicalGlobalObject, attributeName);
}

JSC::JSValue forwardFunctionCallToBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, JSC::JSObject& backingMap, const JSC::Identifier& functionName)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.mapPrototype()->getDirect(vm, functionName);
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);

    JSC::MarkedArgumentBuffer arguments;
    arguments.ensureCapacity(callFrame.argumentCount());
    for (size_t cptr = 0; cptr < callFrame.argumentCount(); ++cptr)
        arguments.append(callFrame.uncheckedArgument(cptr));
    ASSERT(!arguments.hasOverflowed());
    return JSC::call(&lexicalGlobalObject, function, callData, &backingMap, arguments);
}

JSC::JSValue forwardForEachCallToBackingMap(JSDOMGlobalObject& globalObject, JSC::CallFrame& callFrame, JSC::JSObject& mapLike)
{
    auto result = getBackingMap(globalObject, mapLike);
    ASSERT(!result.first);

    auto* function = globalObject.builtinInternalFunctions().jsDOMBindingInternals().m_forEachWrapperFunction.get();
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);

    JSC::MarkedArgumentBuffer arguments;
    arguments.ensureCapacity(callFrame.argumentCount() + 1);
    arguments.append(&result.second.get());
    for (size_t cptr = 0; cptr < callFrame.argumentCount(); ++cptr)
        arguments.append(callFrame.uncheckedArgument(cptr));
    ASSERT(!arguments.hasOverflowed());
    return JSC::call(&globalObject, function, callData, &mapLike, arguments);
}

void DOMMapAdapter::clear()
{
    clearBackingMap(m_lexicalGlobalObject, m_backingMap);
}

}
