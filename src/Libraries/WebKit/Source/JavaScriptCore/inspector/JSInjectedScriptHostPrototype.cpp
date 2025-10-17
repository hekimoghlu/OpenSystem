/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#include "JSInjectedScriptHostPrototype.h"

#include "JSCInlines.h"
#include "JSInjectedScriptHost.h"

namespace Inspector {

using namespace JSC;

static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionSubtype);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionFunctionDetails);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionGetOwnPrivatePropertySymbols);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionGetInternalProperties);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionInternalConstructorName);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIsHTMLAllCollection);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIsPromiseRejectedWithNativeGetterTypeError);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionProxyTargetValue);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakRefTargetValue);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakMapSize);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakMapEntries);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakSetSize);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakSetEntries);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIteratorEntries);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionQueryInstances);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionQueryHolders);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionEvaluateWithScopeExtension);

static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeAttributeEvaluate);
static JSC_DECLARE_HOST_FUNCTION(jsInjectedScriptHostPrototypeAttributeSavedResultAlias);

const ClassInfo JSInjectedScriptHostPrototype::s_info = { "InjectedScriptHost"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSInjectedScriptHostPrototype) };

void JSInjectedScriptHostPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("subtype"_s, jsInjectedScriptHostPrototypeFunctionSubtype, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("functionDetails"_s, jsInjectedScriptHostPrototypeFunctionFunctionDetails, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("getOwnPrivatePropertySymbols"_s, jsInjectedScriptHostPrototypeFunctionGetOwnPrivatePropertySymbols, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("getInternalProperties"_s, jsInjectedScriptHostPrototypeFunctionGetInternalProperties, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("internalConstructorName"_s, jsInjectedScriptHostPrototypeFunctionInternalConstructorName, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("isHTMLAllCollection"_s, jsInjectedScriptHostPrototypeFunctionIsHTMLAllCollection, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("isPromiseRejectedWithNativeGetterTypeError"_s, jsInjectedScriptHostPrototypeFunctionIsPromiseRejectedWithNativeGetterTypeError, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("proxyTargetValue"_s, jsInjectedScriptHostPrototypeFunctionProxyTargetValue, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("weakRefTargetValue"_s, jsInjectedScriptHostPrototypeFunctionWeakRefTargetValue, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("weakMapSize"_s, jsInjectedScriptHostPrototypeFunctionWeakMapSize, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("weakMapEntries"_s, jsInjectedScriptHostPrototypeFunctionWeakMapEntries, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("weakSetSize"_s, jsInjectedScriptHostPrototypeFunctionWeakSetSize, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("weakSetEntries"_s, jsInjectedScriptHostPrototypeFunctionWeakSetEntries, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("iteratorEntries"_s, jsInjectedScriptHostPrototypeFunctionIteratorEntries, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("evaluateWithScopeExtension"_s, jsInjectedScriptHostPrototypeFunctionEvaluateWithScopeExtension, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);

    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("queryInstances"_s, jsInjectedScriptHostPrototypeFunctionQueryInstances, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("queryHolders"_s, jsInjectedScriptHostPrototypeFunctionQueryHolders, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);

    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("evaluate"_s, jsInjectedScriptHostPrototypeAttributeEvaluate, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("savedResultAlias"_s, jsInjectedScriptHostPrototypeAttributeSavedResultAlias, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeAttributeEvaluate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->evaluate(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeAttributeSavedResultAlias, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->savedResultAlias(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionInternalConstructorName, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->internalConstructorName(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIsHTMLAllCollection, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->isHTMLAllCollection(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIsPromiseRejectedWithNativeGetterTypeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->isPromiseRejectedWithNativeGetterTypeError(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionProxyTargetValue, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->proxyTargetValue(callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakRefTargetValue, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->weakRefTargetValue(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakMapSize, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->weakMapSize(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakMapEntries, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->weakMapEntries(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakSetSize, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->weakSetSize(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionWeakSetEntries, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->weakSetEntries(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionIteratorEntries, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->iteratorEntries(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionQueryInstances, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->queryInstances(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionQueryHolders, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->queryHolders(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionEvaluateWithScopeExtension, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->evaluateWithScopeExtension(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionSubtype, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->subtype(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionFunctionDetails, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->functionDetails(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionGetOwnPrivatePropertySymbols, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->getOwnPrivatePropertySymbols(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsInjectedScriptHostPrototypeFunctionGetInternalProperties, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSInjectedScriptHost* castedThis = jsDynamicCast<JSInjectedScriptHost*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->getInternalProperties(globalObject, callFrame));
}

} // namespace Inspector

