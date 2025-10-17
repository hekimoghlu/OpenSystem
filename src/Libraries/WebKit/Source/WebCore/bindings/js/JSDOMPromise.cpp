/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "JSDOMPromise.h"

#include "JSDOMGlobalObject.h"
#include "LocalDOMWindow.h"
#include <JavaScriptCore/BuiltinNames.h>
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/JSNativeStdFunction.h>
#include <JavaScriptCore/JSPromiseConstructor.h>

using namespace JSC;

namespace WebCore {

auto DOMPromise::whenSettled(std::function<void()>&& callback) -> IsCallbackRegistered
{
    return whenPromiseIsSettled(globalObject(), promise(), WTFMove(callback));
}

auto DOMPromise::whenPromiseIsSettled(JSDOMGlobalObject* globalObject, JSC::JSObject* promise, Function<void()>&& callback) -> IsCallbackRegistered
{
    auto& lexicalGlobalObject = *globalObject;
    auto& vm = lexicalGlobalObject.vm();
    JSLockHolder lock(vm);
    auto* handler = JSC::JSNativeStdFunction::create(vm, globalObject, 1, String { }, [callback = WTFMove(callback)] (JSGlobalObject*, CallFrame*) mutable {
        callback();
        return JSC::JSValue::encode(JSC::jsUndefined());
    });

    auto scope = DECLARE_THROW_SCOPE(vm);
    const JSC::Identifier& privateName = vm.propertyNames->builtinNames().thenPrivateName();
    auto thenFunction = promise->get(&lexicalGlobalObject, privateName);

    EXCEPTION_ASSERT(!scope.exception() || vm.hasPendingTerminationException());
    if (scope.exception())
        return IsCallbackRegistered::No;

    ASSERT(thenFunction.isCallable());

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(handler);
    arguments.append(handler);
    ASSERT(!arguments.hasOverflowed());

    auto callData = JSC::getCallData(thenFunction);
    ASSERT(callData.type != JSC::CallData::Type::None);
    call(&lexicalGlobalObject, thenFunction, callData, promise, arguments);

    EXCEPTION_ASSERT(!scope.exception() || vm.hasPendingTerminationException());
    return scope.exception() ? IsCallbackRegistered::No : IsCallbackRegistered::Yes;
}

JSC::JSValue DOMPromise::result() const
{
    return promise()->result(m_globalObject->vm());
}

DOMPromise::Status DOMPromise::status() const
{
    switch (promise()->status(m_globalObject->vm())) {
    case JSC::JSPromise::Status::Pending:
        return Status::Pending;
    case JSC::JSPromise::Status::Fulfilled:
        return Status::Fulfilled;
    case JSC::JSPromise::Status::Rejected:
        return Status::Rejected;
    };
    ASSERT_NOT_REACHED();
    return Status::Rejected;
}

}
