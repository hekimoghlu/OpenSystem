/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#include "InternalWritableStream.h"

#include "Exception.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSArrayBufferViewInlines.h>
#include <JavaScriptCore/JSObjectInlines.h>

namespace WebCore {

static ExceptionOr<JSC::JSValue> invokeWritableStreamFunction(JSC::JSGlobalObject& globalObject, const JSC::Identifier& identifier, const JSC::MarkedArgumentBuffer& arguments)
{
    JSC::VM& vm = globalObject.vm();
    JSC::JSLockHolder lock(vm);

    auto scope = DECLARE_CATCH_SCOPE(vm);

    auto function = globalObject.get(&globalObject, identifier);
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });
    ASSERT(function.isCallable());

    auto callData = JSC::getCallData(function);

    auto result = call(&globalObject, function, callData, JSC::jsUndefined(), arguments);
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    return result;
}

ExceptionOr<JSC::JSValue> InternalWritableStream::writeChunkForBingings(JSC::JSGlobalObject& globalObject, JSC::JSValue chunk)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());
    auto& writerPrivateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().acquireWritableStreamDefaultWriterPrivateName();
    auto writerResult = invokeWritableStreamFunction(globalObject, writerPrivateName, arguments);
    if (UNLIKELY(writerResult.hasException()))
        return writerResult.releaseException();

    arguments.clear();
    arguments.append(writerResult.returnValue());
    arguments.append(chunk);
    ASSERT(!arguments.hasOverflowed());
    auto& writePrivateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().writableStreamDefaultWriterWritePrivateName();
    auto writeResult = invokeWritableStreamFunction(globalObject, writePrivateName, arguments);
    if (UNLIKELY(writeResult.hasException()))
        return writeResult.releaseException();

    arguments.clear();
    arguments.append(writerResult.returnValue());
    ASSERT(!arguments.hasOverflowed());
    auto& releasePrivateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().writableStreamDefaultWriterReleasePrivateName();
    auto releaseResult = invokeWritableStreamFunction(globalObject, releasePrivateName, arguments);
    if (UNLIKELY(releaseResult.hasException()))
        return releaseResult.releaseException();

    return writeResult;
}

ExceptionOr<Ref<InternalWritableStream>> InternalWritableStream::createFromUnderlyingSink(JSDOMGlobalObject& globalObject, JSC::JSValue underlyingSink, JSC::JSValue strategy)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().createInternalWritableStreamFromUnderlyingSinkPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(underlyingSink);
    arguments.append(strategy);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeWritableStreamFunction(globalObject, privateName, arguments);
    if (UNLIKELY(result.hasException()))
        return result.releaseException();

    ASSERT(result.returnValue().isObject());
    return adoptRef(*new InternalWritableStream(globalObject, *result.returnValue().toObject(&globalObject)));
}

Ref<InternalWritableStream> InternalWritableStream::fromObject(JSDOMGlobalObject& globalObject, JSC::JSObject& object)
{
    return adoptRef(*new InternalWritableStream(globalObject, object));
}

bool InternalWritableStream::locked() const
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return false;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().isWritableStreamLockedPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeWritableStreamFunction(*globalObject, privateName, arguments);
    if (scope.exception())
        scope.clearException();

    return result.hasException() ? false : result.returnValue().isTrue();
}

void InternalWritableStream::lock()
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().acquireWritableStreamDefaultWriterPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    invokeWritableStreamFunction(*globalObject, privateName, arguments);
    if (UNLIKELY(scope.exception()))
        scope.clearException();
}

JSC::JSValue InternalWritableStream::abortForBindings(JSC::JSGlobalObject& globalObject, JSC::JSValue reason)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().writableStreamAbortForBindingsPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    arguments.append(reason);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeWritableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

JSC::JSValue InternalWritableStream::closeForBindings(JSC::JSGlobalObject& globalObject)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().writableStreamCloseForBindingsPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeWritableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

void InternalWritableStream::closeIfPossible()
{
    auto* globalObject = this->globalObject();
    if (!globalObject)
        return;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());

    auto* clientData = static_cast<JSVMClientData*>(globalObject->vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().writableStreamCloseIfPossiblePrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    invokeWritableStreamFunction(*globalObject, privateName, arguments);
    if (UNLIKELY(scope.exception()))
        scope.clearException();
}

JSC::JSValue InternalWritableStream::getWriter(JSC::JSGlobalObject& globalObject)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().writableStreamInternalsBuiltins().acquireWritableStreamDefaultWriterPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(guardedObject());
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeWritableStreamFunction(globalObject, privateName, arguments);
    if (result.hasException())
        return { };

    return result.returnValue();
}

}
