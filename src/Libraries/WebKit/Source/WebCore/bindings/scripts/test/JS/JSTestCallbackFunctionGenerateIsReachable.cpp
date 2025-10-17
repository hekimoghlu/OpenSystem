/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "JSTestCallbackFunctionGenerateIsReachable.h"

#include "ContextDestructionObserverInlines.h"
#include "JSDOMConvertNumbers.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMExceptionHandling.h"
#include "ScriptExecutionContext.h"


namespace WebCore {
using namespace JSC;

JSTestCallbackFunctionGenerateIsReachable::JSTestCallbackFunctionGenerateIsReachable(JSObject* callback, JSDOMGlobalObject* globalObject)
    : TestCallbackFunctionGenerateIsReachable(globalObject->scriptExecutionContext())
    , m_data(new JSCallbackData(callback, globalObject, globalObject->scriptExecutionContext()))
{
}

JSTestCallbackFunctionGenerateIsReachable::~JSTestCallbackFunctionGenerateIsReachable()
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

CallbackResult<typename IDLDOMString::CallbackReturnType> JSTestCallbackFunctionGenerateIsReachable::handleEvent(typename IDLLong::ParameterType argument)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestCallbackFunctionGenerateIsReachable> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLLong>(argument));
    ASSERT(!args.hasOverflowed());

    NakedPtr<JSC::Exception> returnedException;
    auto jsResult = m_data->invokeCallback(thisValue, args, JSCallbackData::CallbackType::Function, Identifier(), returnedException);
    if (returnedException) {
        UNUSED_PARAM(lexicalGlobalObject);
        reportException(m_data->callback()->globalObject(), returnedException);
        return CallbackResultType::ExceptionThrown;
     }

    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto returnValue = convert<IDLDOMString>(lexicalGlobalObject, jsResult);
    if (UNLIKELY(returnValue.hasException(throwScope)))
        return CallbackResultType::ExceptionThrown;
    return { returnValue.releaseReturnValue() };
}

CallbackResult<typename IDLDOMString::CallbackReturnType> JSTestCallbackFunctionGenerateIsReachable::handleEventRethrowingException(typename IDLLong::ParameterType argument)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestCallbackFunctionGenerateIsReachable> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLLong>(argument));
    ASSERT(!args.hasOverflowed());

    NakedPtr<JSC::Exception> returnedException;
    auto jsResult = m_data->invokeCallback(thisValue, args, JSCallbackData::CallbackType::Function, Identifier(), returnedException);
    if (returnedException) {
        auto throwScope = DECLARE_THROW_SCOPE(vm);
        throwException(&lexicalGlobalObject, throwScope, returnedException);
        return CallbackResultType::ExceptionThrown;
     }

    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto returnValue = convert<IDLDOMString>(lexicalGlobalObject, jsResult);
    if (UNLIKELY(returnValue.hasException(throwScope)))
        return CallbackResultType::ExceptionThrown;
    return { returnValue.releaseReturnValue() };
}

JSC::JSValue toJS(TestCallbackFunctionGenerateIsReachable& impl)
{
    if (!static_cast<JSTestCallbackFunctionGenerateIsReachable&>(impl).callbackData())
        return jsNull();

    return static_cast<JSTestCallbackFunctionGenerateIsReachable&>(impl).callbackData()->callback();
}

} // namespace WebCore
