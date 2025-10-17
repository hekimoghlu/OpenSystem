/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "JSCallbackData.h"

#include "Document.h"
#include "JSDOMBinding.h"
#include "JSExecState.h"
#include "JSExecStateInstrumentation.h"
#include <JavaScriptCore/Exception.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSCallbackData);

// https://webidl.spec.whatwg.org/#call-a-user-objects-operation
JSValue JSCallbackData::invokeCallback(JSDOMGlobalObject& globalObject, JSObject* callback, JSValue thisValue, MarkedArgumentBuffer& args, CallbackType method, PropertyName functionName, NakedPtr<JSC::Exception>& returnedException)
{
    ASSERT(callback);

    JSGlobalObject* lexicalGlobalObject = &globalObject;
    VM& vm = lexicalGlobalObject->vm();

    auto scope = DECLARE_CATCH_SCOPE(vm);

    JSValue function;
    CallData callData;

    if (method != CallbackType::Object) {
        function = callback;
        callData = JSC::getCallData(callback);
    }
    if (callData.type == CallData::Type::None) {
        if (method == CallbackType::Function) {
            returnedException = JSC::Exception::create(vm, createTypeError(lexicalGlobalObject));
            return JSValue();
        }

        ASSERT(!functionName.isNull());
        function = callback->get(lexicalGlobalObject, functionName);
        if (UNLIKELY(scope.exception())) {
            returnedException = scope.exception();
            scope.clearException();
            return JSValue();
        }

        callData = JSC::getCallData(function);
        if (callData.type == CallData::Type::None) {
            returnedException = JSC::Exception::create(vm, createTypeError(lexicalGlobalObject, makeString('\'', String(functionName.uid()), "' property of callback interface should be callable"_s)));
            return JSValue();
        }

        thisValue = callback;
    }

    ASSERT(!function.isEmpty());
    ASSERT(callData.type != CallData::Type::None);

    ScriptExecutionContext* context = globalObject.scriptExecutionContext();
    // We will fail to get the context if the frame has been detached.
    if (!context)
        return JSValue();

    JSExecState::instrumentFunction(context, callData);

    returnedException = nullptr;
    JSValue result = JSExecState::profiledCall(lexicalGlobalObject, JSC::ProfilingReason::Other, function, callData, thisValue, args, returnedException);

    InspectorInstrumentation::didCallFunction(context);

    return result;
}

template<typename Visitor>
void JSCallbackData::visitJSFunction(Visitor& visitor)
{
    visitor.append(m_callback);
}

template void JSCallbackData::visitJSFunction(JSC::AbstractSlotVisitor&);
template void JSCallbackData::visitJSFunction(JSC::SlotVisitor&);

} // namespace WebCore
