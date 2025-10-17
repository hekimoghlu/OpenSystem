/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#pragma once

#include "IDLTypes.h"
#include "JSDOMConvertBase.h"
#include "JSDOMPromise.h"
#include "WorkerGlobalScope.h"

namespace WebCore {

template<typename T> struct Converter<IDLPromise<T>> : DefaultConverter<IDLPromise<T>> {
    using ReturnType = Ref<DOMPromise>;
    using Result = ConversionResult<IDLPromise<T>>;

    // https://webidl.spec.whatwg.org/#es-promise
    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);
        auto* globalObject = JSC::jsDynamicCast<JSDOMGlobalObject*>(&lexicalGlobalObject);
        RELEASE_ASSERT(globalObject);

        // 1. Let resolve be the original value of %Promise%.resolve.
        // 2. Let promise be the result of calling resolve with %Promise% as the this value and V as the single argument value.
        auto* promise = JSC::JSPromise::resolvedPromise(globalObject, value);
        if (scope.exception()) {
            auto* scriptExecutionContext = globalObject->scriptExecutionContext();
            if (auto* globalScope = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext)) {
                auto* scriptController = globalScope->script();
                bool terminatorCausedException = vm.isTerminationException(scope.exception());
                if (terminatorCausedException || (scriptController && scriptController->isTerminatingExecution())) {
                    scriptController->forbidExecution();
                    return Result::exception();
                }
            }
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }
        ASSERT(promise);

        // 3. Return the IDL promise type value that is a reference to the same object as promise.
        return DOMPromise::create(*globalObject, *promise);
    }
};

template<typename T> struct Converter<IDLPromiseIgnoringSuspension<T>> : public Converter<IDLPromise<T>> { };

template<typename T> struct JSConverter<IDLPromise<T>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    static JSC::JSValue convert(JSC::JSGlobalObject&, JSDOMGlobalObject&, DOMPromise& promise)
    {
        return promise.promise();
    }

    static JSC::JSValue convert(JSC::JSGlobalObject&, JSDOMGlobalObject&, const RefPtr<DOMPromise>& promise)
    {
        return promise->promise();
    }

    template<template<typename> class U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, U<T>& promiseProxy)
    {
        return promiseProxy.promise(lexicalGlobalObject, globalObject);
    }
};

template<typename T> struct JSConverter<IDLPromiseIgnoringSuspension<T>> : public JSConverter<IDLPromise<T>> {
    static JSC::JSValue convert(JSC::JSGlobalObject&, JSDOMGlobalObject&, DOMPromise& promise)
    {
        return promise.guardedObject();
    }

    static JSC::JSValue convert(JSC::JSGlobalObject&, JSDOMGlobalObject&, const RefPtr<DOMPromise>& promise)
    {
        return promise->guardedObject();
    }
};

} // namespace WebCore
