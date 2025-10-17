/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include "JSDOMSetLike.h"

#include "WebCoreJSBuiltinInternals.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/JSSet.h>
#include <JavaScriptCore/JSSetInlines.h>
#include <JavaScriptCore/VMTrapsInlines.h>

namespace WebCore {

void DOMSetAdapter::clear()
{
    clearBackingSet(m_lexicalGlobalObject, m_backingSet);
}

std::pair<bool, std::reference_wrapper<JSC::JSObject>> getBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& setLike)
{
    auto& vm = lexicalGlobalObject.vm();
    auto backingSet = setLike.getDirect(vm, builtinNames(vm).backingSetPrivateName());
    if (!backingSet) {
        auto& vm = lexicalGlobalObject.vm();
        backingSet = JSC::JSSet::create(vm, lexicalGlobalObject.setStructure());
        setLike.putDirect(vm, builtinNames(vm).backingSetPrivateName(), backingSet, enumToUnderlyingType(JSC::PropertyAttribute::DontEnum));
        return { true, *JSC::asObject(backingSet) };
    }
    return { false, *JSC::asObject(backingSet) };
}

void clearBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingSet)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.jsSetPrototype()->getDirect(vm, vm.propertyNames->builtinNames().clearPrivateName());
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);
    JSC::MarkedArgumentBuffer arguments;
    JSC::call(&lexicalGlobalObject, function, callData, &backingSet, arguments);
}

void addToBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingSet, JSC::JSValue item)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.jsSetPrototype()->getDirect(vm, vm.propertyNames->builtinNames().addPrivateName());
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);
    JSC::MarkedArgumentBuffer arguments;
    arguments.append(item);
    ASSERT(!arguments.hasOverflowed());
    JSC::call(&lexicalGlobalObject, function, callData, &backingSet, arguments);
}

JSC::JSValue forwardAttributeGetterToBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingSet, const JSC::Identifier& attributeName)
{
    return backingSet.get(&lexicalGlobalObject, attributeName);
}

JSC::JSValue forwardFunctionCallToBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, JSC::JSObject& backingSet, const JSC::Identifier& functionName)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto function = lexicalGlobalObject.jsSetPrototype()->getDirect(vm, functionName);
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);
    JSC::MarkedArgumentBuffer arguments;
    arguments.ensureCapacity(callFrame.argumentCount());
    for (size_t cptr = 0; cptr < callFrame.argumentCount(); ++cptr)
        arguments.append(callFrame.uncheckedArgument(cptr));
    ASSERT(!arguments.hasOverflowed());
    return JSC::call(&lexicalGlobalObject, function, callData, &backingSet, arguments);
}

JSC::JSValue forwardForEachCallToBackingSet(JSDOMGlobalObject& globalObject, JSC::CallFrame& callFrame, JSC::JSObject& setLike)
{
    auto result = getBackingSet(globalObject, setLike);
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
    return JSC::call(&globalObject, function, callData, &setLike, arguments);
}

}
