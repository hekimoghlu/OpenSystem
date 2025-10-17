/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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

#if ENABLE(TEST_CONDITIONAL)

#include "JSTestVoidCallbackFunction.h"

#include "ContextDestructionObserverInlines.h"
#include "JSDOMConvertBase.h"
#include "JSDOMConvertBoolean.h"
#include "JSDOMConvertBufferSource.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNumbers.h"
#include "JSDOMConvertSerializedScriptValue.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMConvertUnion.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObject.h"
#include "JSTestNode.h"
#include "ScriptExecutionContext.h"
#include "SerializedScriptValue.h"


namespace WebCore {
using namespace JSC;

JSTestVoidCallbackFunction::JSTestVoidCallbackFunction(JSObject* callback, JSDOMGlobalObject* globalObject)
    : TestVoidCallbackFunction(globalObject->scriptExecutionContext())
    , m_data(new JSCallbackData(callback, globalObject, this))
{
}

JSTestVoidCallbackFunction::~JSTestVoidCallbackFunction()
{
    SUPPRESS_UNCOUNTED_LOCAL ScriptExecutionContext* context = scriptExecutionContext();
    // When the context is destroyed, all tasks with a reference to a callback
    // should be deleted. So if the context is 0, we are on the context thread.
    // We can't use RefPtr here since ScriptExecutionContext is not thread safe ref counted.
    if (!context || context->isContextThread())
        delete m_data;
    else
        context->postTask(DeleteCallbackDataTask(m_data));
#ifndef NDEBUG
    m_data = nullptr;
#endif
}

CallbackResult<typename IDLUndefined::CallbackReturnType> JSTestVoidCallbackFunction::handleEvent(typename IDLFloat32Array::ParameterType arrayParam, typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strArg, typename IDLBoolean::ParameterType boolParam, typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestVoidCallbackFunction> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLFloat32Array>(lexicalGlobalObject, globalObject, arrayParam));
    args.append(toJS<IDLSerializedScriptValue<SerializedScriptValue>>(lexicalGlobalObject, globalObject, srzParam));
    args.append(toJS<IDLDOMString>(lexicalGlobalObject, strArg));
    args.append(toJS<IDLBoolean>(boolParam));
    args.append(toJS<IDLLong>(longParam));
    args.append(toJS<IDLInterface<TestNode>>(lexicalGlobalObject, globalObject, testNodeParam));
    ASSERT(!args.hasOverflowed());

    NakedPtr<JSC::Exception> returnedException;
    m_data->invokeCallback(thisValue, args, JSCallbackData::CallbackType::Function, Identifier(), returnedException);
    if (returnedException) {
        UNUSED_PARAM(lexicalGlobalObject);
        reportException(m_data->callback()->globalObject(), returnedException);
        return CallbackResultType::ExceptionThrown;
     }

    return { };
}

CallbackResult<typename IDLUndefined::CallbackReturnType> JSTestVoidCallbackFunction::handleEventRethrowingException(typename IDLFloat32Array::ParameterType arrayParam, typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strArg, typename IDLBoolean::ParameterType boolParam, typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestVoidCallbackFunction> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLFloat32Array>(lexicalGlobalObject, globalObject, arrayParam));
    args.append(toJS<IDLSerializedScriptValue<SerializedScriptValue>>(lexicalGlobalObject, globalObject, srzParam));
    args.append(toJS<IDLDOMString>(lexicalGlobalObject, strArg));
    args.append(toJS<IDLBoolean>(boolParam));
    args.append(toJS<IDLLong>(longParam));
    args.append(toJS<IDLInterface<TestNode>>(lexicalGlobalObject, globalObject, testNodeParam));
    ASSERT(!args.hasOverflowed());

    NakedPtr<JSC::Exception> returnedException;
    m_data->invokeCallback(thisValue, args, JSCallbackData::CallbackType::Function, Identifier(), returnedException);
    if (returnedException) {
        auto throwScope = DECLARE_THROW_SCOPE(vm);
        throwException(&lexicalGlobalObject, throwScope, returnedException);
        return CallbackResultType::ExceptionThrown;
     }

    return { };
}

void JSTestVoidCallbackFunction::visitJSFunction(JSC::AbstractSlotVisitor& visitor)
{
    m_data->visitJSFunction(visitor);
}

void JSTestVoidCallbackFunction::visitJSFunction(JSC::SlotVisitor& visitor)
{
    m_data->visitJSFunction(visitor);
}

JSC::JSValue toJS(TestVoidCallbackFunction& impl)
{
    if (!static_cast<JSTestVoidCallbackFunction&>(impl).callbackData())
        return jsNull();

    return static_cast<JSTestVoidCallbackFunction&>(impl).callbackData()->callback();
}

} // namespace WebCore

#endif // ENABLE(TEST_CONDITIONAL)
