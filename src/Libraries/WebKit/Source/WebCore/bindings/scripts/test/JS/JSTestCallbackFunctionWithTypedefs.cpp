/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "JSTestCallbackFunctionWithTypedefs.h"

#include "ContextDestructionObserverInlines.h"
#include "JSDOMConvertBase.h"
#include "JSDOMConvertNullable.h"
#include "JSDOMConvertNumbers.h"
#include "JSDOMConvertSequences.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObject.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/JSArray.h>


namespace WebCore {
using namespace JSC;

JSTestCallbackFunctionWithTypedefs::JSTestCallbackFunctionWithTypedefs(JSObject* callback, JSDOMGlobalObject* globalObject)
    : TestCallbackFunctionWithTypedefs(globalObject->scriptExecutionContext())
    , m_data(new JSCallbackData(callback, globalObject, this))
{
}

JSTestCallbackFunctionWithTypedefs::~JSTestCallbackFunctionWithTypedefs()
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

CallbackResult<typename IDLUndefined::CallbackReturnType> JSTestCallbackFunctionWithTypedefs::handleEvent(typename IDLSequence<IDLNullable<IDLLong>>::ParameterType sequenceArg, typename IDLLong::ParameterType longArg)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestCallbackFunctionWithTypedefs> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLSequence<IDLNullable<IDLLong>>>(lexicalGlobalObject, globalObject, sequenceArg));
    args.append(toJS<IDLLong>(longArg));
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

CallbackResult<typename IDLUndefined::CallbackReturnType> JSTestCallbackFunctionWithTypedefs::handleEventRethrowingException(typename IDLSequence<IDLNullable<IDLLong>>::ParameterType sequenceArg, typename IDLLong::ParameterType longArg)
{
    if (!canInvokeCallback())
        return CallbackResultType::UnableToExecute;

    Ref<JSTestCallbackFunctionWithTypedefs> protectedThis(*this);

    auto& globalObject = *m_data->globalObject();
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject.vm();

    JSLockHolder lock(vm);
    auto& lexicalGlobalObject = globalObject;
    JSValue thisValue = jsUndefined();
    MarkedArgumentBuffer args;
    args.append(toJS<IDLSequence<IDLNullable<IDLLong>>>(lexicalGlobalObject, globalObject, sequenceArg));
    args.append(toJS<IDLLong>(longArg));
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

void JSTestCallbackFunctionWithTypedefs::visitJSFunction(JSC::AbstractSlotVisitor& visitor)
{
    m_data->visitJSFunction(visitor);
}

void JSTestCallbackFunctionWithTypedefs::visitJSFunction(JSC::SlotVisitor& visitor)
{
    m_data->visitJSFunction(visitor);
}

JSC::JSValue toJS(TestCallbackFunctionWithTypedefs& impl)
{
    if (!static_cast<JSTestCallbackFunctionWithTypedefs&>(impl).callbackData())
        return jsNull();

    return static_cast<JSTestCallbackFunctionWithTypedefs&>(impl).callbackData()->callback();
}

} // namespace WebCore
