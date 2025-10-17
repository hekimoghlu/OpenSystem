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
#include "InternalReadableStream.h"

#include "JSDOMConvertObject.h"
#include "JSDOMConvertSequences.h"
#include "JSDOMException.h"
#include "JSReadableStreamSink.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSObjectInlines.h>

namespace WebCore {

static ExceptionOr<JSC::JSValue> invokeReadableStreamFunction(JSC::JSGlobalObject& globalObject, const JSC::Identifier& identifier, const JSC::MarkedArgumentBuffer& arguments)
{
    JSC::VM& vm = globalObject.vm();
    JSC::JSLockHolder lock(vm);

    auto scope = DECLARE_CATCH_SCOPE(vm);

    auto function = globalObject.get(&globalObject, identifier);
    ASSERT(!!scope.exception() || function.isCallable());
    scope.assertNoExceptionExceptTermination();
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    auto callData = JSC::getCallData(function);

    auto result = call(&globalObject, function, callData, JSC::jsUndefined(), arguments);
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    return result;
}

ExceptionOr<Ref<InternalReadableStream>> InternalReadableStream::createFromUnderlyingSource(JSDOMGlobalObject& globalObject, JSC::JSValue underlyingSource, JSC::JSValue strategy)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().createInternalReadableStreamFromUnderlyingSourcePrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(underlyingSource);
    arguments.append(strategy);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (UNLIKELY(result.hasException()))
        return result.releaseException();

    ASSERT(result.returnValue().isObject());
    return adoptRef(*new InternalReadableStream(globalObject, *result.returnValue().toObject(&globalObject)));
}

Ref<InternalReadableStream> InternalReadableStream::fromObject(JSDOMGlobalObject& globalObject, JSC::JSObject& object)
{
    return adoptRef(*new InternalReadableStream(globalObject, object));
}

bool InternalReadableStream::isLocked() const
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return false;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().isReadableStreamLockedPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(*globalObject, privateName, arguments);
    if (scope.exception())
        scope.clearException();

    return result.hasException() ? false : result.returnValue().isTrue();
}

bool InternalReadableStream::isDisturbed() const
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return false;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().isReadableStreamDisturbedPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(*globalObject, privateName, arguments);
    if (scope.exception())
        scope.clearException();

    return result.hasException() ? false : result.returnValue().isTrue();
}

void InternalReadableStream::cancel(Exception&& exception)
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());
    JSC::JSLockHolder lock(globalObject->vm());
    cancel(*globalObject, toJSNewlyCreated(globalObject, JSC::jsCast<JSDOMGlobalObject*>(globalObject), DOMException::create(WTFMove(exception))), Use::Private);
    if (UNLIKELY(scope.exception()))
        scope.clearException();
}

void InternalReadableStream::lock()
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().acquireReadableStreamDefaultReaderPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    invokeReadableStreamFunction(*globalObject, privateName, arguments);
    if (UNLIKELY(scope.exception()))
        scope.clearException();
}

void InternalReadableStream::pipeTo(ReadableStreamSink& sink)
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());
    JSC::JSLockHolder lock(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamPipeToPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(toJS(globalObject, globalObject, sink));
    ASSERT(!arguments.hasOverflowed());

    invokeReadableStreamFunction(*globalObject, privateName, arguments);
    if (UNLIKELY(scope.exception()))
        scope.clearException();
}

ExceptionOr<std::pair<Ref<InternalReadableStream>, Ref<InternalReadableStream>>> InternalReadableStream::tee(bool shouldClone)
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return Exception { ExceptionCode::InvalidStateError };

    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    auto result = tee(*globalObject, shouldClone);
    if (UNLIKELY(scope.exception()))
        return Exception { ExceptionCode::ExistingExceptionError };

    auto resultsConversionResult = convert<IDLSequence<IDLObject>>(*globalObject, result);
    if (UNLIKELY(resultsConversionResult.hasException(scope)))
        return Exception { ExceptionCode::ExistingExceptionError };

    auto results = resultsConversionResult.releaseReturnValue();
    ASSERT(results.size() == 2);

    auto& jsDOMGlobalObject = *JSC::jsCast<JSDOMGlobalObject*>(globalObject);
    return std::make_pair(InternalReadableStream::fromObject(jsDOMGlobalObject, *results[0].get()), InternalReadableStream::fromObject(jsDOMGlobalObject, *results[1].get()));
}

JSC::JSValue InternalReadableStream::cancel(JSC::JSGlobalObject& globalObject, JSC::JSValue reason, Use use)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& names = clientData->builtinFunctions().readableStreamInternalsBuiltins();
    auto& privateName = use == Use::Bindings ? names.readableStreamCancelForBindingsPrivateName() : names.readableStreamCancelPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(reason);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

JSC::JSValue InternalReadableStream::getReader(JSC::JSGlobalObject& globalObject, JSC::JSValue options)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamGetReaderForBindingsPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(options);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

JSC::JSValue InternalReadableStream::pipeTo(JSC::JSGlobalObject& globalObject, JSC::JSValue streams, JSC::JSValue options)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamPipeToForBindingsPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(streams);
    arguments.append(options);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

JSC::JSValue InternalReadableStream::pipeThrough(JSC::JSGlobalObject& globalObject, JSC::JSValue dest, JSC::JSValue options)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamPipeThroughForBindingsPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(dest);
    arguments.append(options);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

JSC::JSValue InternalReadableStream::tee(JSC::JSGlobalObject& globalObject, bool shouldClone)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamTeePrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(shouldClone ? JSC::JSValue(JSC::JSValue::JSTrue) : JSC::JSValue(JSC::JSValue::JSFalse));
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeReadableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

}
