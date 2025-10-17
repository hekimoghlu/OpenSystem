/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include "ReadableStreamDefaultController.h"

#include "WebCoreJSClientData.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/IdentifierInlines.h>
#include <JavaScriptCore/JSObjectInlines.h>

namespace WebCore {

static bool invokeReadableStreamDefaultControllerFunction(JSC::JSGlobalObject& lexicalGlobalObject, const JSC::Identifier& identifier, const JSC::MarkedArgumentBuffer& arguments)
{
    JSC::VM& vm = lexicalGlobalObject.vm();
    JSC::JSLockHolder lock(vm);

    auto scope = DECLARE_CATCH_SCOPE(vm);
    auto function = lexicalGlobalObject.get(&lexicalGlobalObject, identifier);

    EXCEPTION_ASSERT(!scope.exception() || vm.hasPendingTerminationException());
    RETURN_IF_EXCEPTION(scope, false);

    ASSERT(function.isCallable());

    auto callData = JSC::getCallData(function);
    call(&lexicalGlobalObject, function, callData, JSC::jsUndefined(), arguments);
    EXCEPTION_ASSERT(!scope.exception() || vm.hasPendingTerminationException());
    return !scope.exception();
}

void ReadableStreamDefaultController::close()
{
    JSC::MarkedArgumentBuffer arguments;
    arguments.append(&jsController());
    ASSERT(!arguments.hasOverflowed());

    auto* clientData = static_cast<JSVMClientData*>(globalObject().vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamDefaultControllerClosePrivateName();

    invokeReadableStreamDefaultControllerFunction(globalObject(), privateName, arguments);
}


void ReadableStreamDefaultController::error(const Exception& exception)
{
    JSC::JSGlobalObject& lexicalGlobalObject = this->globalObject();
    auto& vm = lexicalGlobalObject.vm();
    JSC::JSLockHolder lock(vm);
    auto scope = DECLARE_CATCH_SCOPE(vm);
    auto value = createDOMException(&lexicalGlobalObject, exception.code(), exception.message());

    if (UNLIKELY(scope.exception())) {
        ASSERT(vm.hasPendingTerminationException());
        return;
    }

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(&jsController());
    arguments.append(value);
    ASSERT(!arguments.hasOverflowed());

    auto* clientData = static_cast<JSVMClientData*>(vm.clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamDefaultControllerErrorPrivateName();

    invokeReadableStreamDefaultControllerFunction(globalObject(), privateName, arguments);
}

bool ReadableStreamDefaultController::enqueue(JSC::JSValue value)
{
    JSC::JSGlobalObject& lexicalGlobalObject = this->globalObject();
    auto& vm = lexicalGlobalObject.vm();
    JSC::JSLockHolder lock(vm);

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(&jsController());
    arguments.append(value);
    ASSERT(!arguments.hasOverflowed());

    auto* clientData = static_cast<JSVMClientData*>(lexicalGlobalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().readableStreamInternalsBuiltins().readableStreamDefaultControllerEnqueuePrivateName();

    return invokeReadableStreamDefaultControllerFunction(globalObject(), privateName, arguments);
}

bool ReadableStreamDefaultController::enqueue(RefPtr<JSC::ArrayBuffer>&& buffer)
{
    if (!buffer) {
        error(Exception { ExceptionCode::OutOfMemoryError });
        return false;
    }

    JSC::JSGlobalObject& lexicalGlobalObject = this->globalObject();
    auto& vm = lexicalGlobalObject.vm();
    JSC::JSLockHolder lock(vm);
    auto scope = DECLARE_CATCH_SCOPE(vm);
    auto length = buffer->byteLength();
    auto chunk = JSC::Uint8Array::create(WTFMove(buffer), 0, length);
    auto value = toJS(&lexicalGlobalObject, &lexicalGlobalObject, chunk.get());

    EXCEPTION_ASSERT(!scope.exception() || vm.hasPendingTerminationException());
    RETURN_IF_EXCEPTION(scope, false);

    return enqueue(value);
}

} // namespace WebCore
