/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#include "JSJavaScriptCallFramePrototype.h"

#include "JSCInlines.h"
#include "JSJavaScriptCallFrame.h"

namespace Inspector {

using namespace JSC;

// Functions.
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFramePrototypeFunctionEvaluateWithScopeExtension);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFramePrototypeFunctionScopeDescriptions);

// Attributes.
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeCaller);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeSourceID);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeLine);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeColumn);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeFunctionName);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeScopeChain);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeThisObject);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeType);
static JSC_DECLARE_HOST_FUNCTION(jsJavaScriptCallFrameIsTailDeleted);

const ClassInfo JSJavaScriptCallFramePrototype::s_info = { "JavaScriptCallFrame"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSJavaScriptCallFramePrototype) };

void JSJavaScriptCallFramePrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("evaluateWithScopeExtension"_s, jsJavaScriptCallFramePrototypeFunctionEvaluateWithScopeExtension, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("scopeDescriptions"_s, jsJavaScriptCallFramePrototypeFunctionScopeDescriptions, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Private);

    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("caller"_s, jsJavaScriptCallFrameAttributeCaller, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("sourceID"_s, jsJavaScriptCallFrameAttributeSourceID, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("line"_s, jsJavaScriptCallFrameAttributeLine, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("column"_s, jsJavaScriptCallFrameAttributeColumn, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("functionName"_s, jsJavaScriptCallFrameAttributeFunctionName, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("scopeChain"_s, jsJavaScriptCallFrameAttributeScopeChain, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("thisObject"_s, jsJavaScriptCallFrameAttributeThisObject, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("type"_s, jsJavaScriptCallFrameAttributeType, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    JSC_NATIVE_GETTER_WITHOUT_TRANSITION("isTailDeleted"_s, jsJavaScriptCallFrameIsTailDeleted, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFramePrototypeFunctionEvaluateWithScopeExtension, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->evaluateWithScopeExtension(globalObject, callFrame));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFramePrototypeFunctionScopeDescriptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->scopeDescriptions(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeCaller, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->caller(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeSourceID, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->sourceID(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeLine, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->line(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeColumn, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->column(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeFunctionName, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->functionName(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeScopeChain, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->scopeChain(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeThisObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->thisObject(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameAttributeType, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->type(globalObject));
}

JSC_DEFINE_HOST_FUNCTION(jsJavaScriptCallFrameIsTailDeleted, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSJavaScriptCallFrame* castedThis = jsDynamicCast<JSJavaScriptCallFrame*>(thisValue);
    if (!castedThis)
        return throwVMTypeError(globalObject, scope);

    return JSValue::encode(castedThis->isTailDeleted(globalObject));
}

} // namespace Inspector
