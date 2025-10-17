/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include "ScriptFunctionCall.h"

#include "JSCInlines.h"
#include "JSLock.h"
#include <wtf/text/WTFString.h>

namespace Inspector {

using namespace JSC;

void ScriptCallArgumentHandler::appendArgument(const String& argument)
{
    VM& vm = m_globalObject->vm();
    JSLockHolder lock(vm);
    m_arguments.append(jsString(vm, argument));
}

void ScriptCallArgumentHandler::appendArgument(const char* argument)
{
    VM& vm = m_globalObject->vm();
    JSLockHolder lock(vm);
    m_arguments.append(jsString(vm, String::fromLatin1(argument)));
}

void ScriptCallArgumentHandler::appendArgument(JSValue argument)
{
    m_arguments.append(argument);
}

void ScriptCallArgumentHandler::appendArgument(long argument)
{
    JSLockHolder lock(m_globalObject);
    m_arguments.append(jsNumber(argument));
}

void ScriptCallArgumentHandler::appendArgument(long long argument)
{
    JSLockHolder lock(m_globalObject);
    m_arguments.append(jsNumber(argument));
}

void ScriptCallArgumentHandler::appendArgument(unsigned int argument)
{
    JSLockHolder lock(m_globalObject);
    m_arguments.append(jsNumber(argument));
}

void ScriptCallArgumentHandler::appendArgument(uint64_t argument)
{
    JSLockHolder lock(m_globalObject);
    m_arguments.append(jsNumber(argument));
}

void ScriptCallArgumentHandler::appendArgument(int argument)
{
    JSLockHolder lock(m_globalObject);
    m_arguments.append(jsNumber(argument));
}

void ScriptCallArgumentHandler::appendArgument(bool argument)
{
    m_arguments.append(jsBoolean(argument));
}

ScriptFunctionCall::ScriptFunctionCall(JSC::JSGlobalObject* globalObject, JSC::JSObject* thisObject, const String& name, ScriptFunctionCallHandler callHandler)
    : ScriptCallArgumentHandler(globalObject)
    , m_callHandler(callHandler)
    , m_thisObject(globalObject->vm(), thisObject)
    , m_name(name)
{
}

Expected<JSValue, NakedPtr<Exception>> ScriptFunctionCall::call()
{
    JSObject* thisObject = m_thisObject.get();

    VM& vm = m_globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_CATCH_SCOPE(vm);

    auto makeExceptionResult = [&] (Exception* exception) -> Expected<JSValue, NakedPtr<Exception>> {
        // Do not treat a terminated execution exception as having an exception. Just treat it as an empty result.
        if (!vm.isTerminationException(exception))
            return makeUnexpected(exception);
        return { };
    };

    JSValue function = thisObject->get(m_globalObject, Identifier::fromString(vm, m_name));
    Exception* exception = scope.exception();
    if (UNLIKELY(exception)) {
        scope.clearException();
        return makeExceptionResult(exception);
    }

    auto callData = JSC::getCallData(function);
    if (callData.type == CallData::Type::None)
        return { };

    JSValue result;
    NakedPtr<Exception> uncaughtException;
    if (m_callHandler)
        result = m_callHandler(m_globalObject, function, callData, thisObject, m_arguments, uncaughtException);
    else
        result = JSC::call(m_globalObject, function, callData, thisObject, m_arguments, uncaughtException);

    if (uncaughtException)
        return makeExceptionResult(uncaughtException);

    return result;
}

} // namespace Inspector
