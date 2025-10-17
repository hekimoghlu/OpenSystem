/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

#include "JSDOMOperation.h"
#include "JSDOMPromiseDeferred.h"

namespace WebCore {

template<typename JSClass>
class IDLOperationReturningPromise {
public:
    using ClassParameter = JSClass*;
    using Operation = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*, ClassParameter, Ref<DeferredPromise>&&);
    using StaticOperation = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*, Ref<DeferredPromise>&&);

    template<Operation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::RejectPromise>
    static JSC::EncodedJSValue call(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char* operationName)
    {
        return JSC::JSValue::encode(callPromiseFunction(lexicalGlobalObject, callFrame, [&operationName] (JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, Ref<DeferredPromise>&& promise) {
            auto* thisObject = IDLOperation<JSClass>::cast(lexicalGlobalObject, callFrame);
            if constexpr (shouldThrow != CastedThisErrorBehavior::Assert) {
                if (UNLIKELY(!thisObject))
                    return rejectPromiseWithThisTypeError(promise.get(), JSClass::info()->className, operationName);
            } else {
                UNUSED_PARAM(operationName);
                ASSERT(thisObject);
            }

            ASSERT_GC_OBJECT_INHERITS(thisObject, JSClass::info());
            
            // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject and thisObject.
            return operation(&lexicalGlobalObject, &callFrame, thisObject, WTFMove(promise));
        }));
    }

    using Operation2 = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*, ClassParameter, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);
    template<Operation2 operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::RejectPromise>
    static JSC::EncodedJSValue callReturningPromisePair(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char* operationName)
    {
        return callPromisePairFunction(lexicalGlobalObject, callFrame, [&operationName] (JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, Ref<DeferredPromise>&& promise, Ref<DeferredPromise>&& promise2) {
            auto* thisObject = IDLOperation<JSClass>::cast(lexicalGlobalObject, callFrame);
            if constexpr (shouldThrow != CastedThisErrorBehavior::Assert) {
                if (UNLIKELY(!thisObject))
                    return rejectPromiseWithThisTypeError(promise.get(), JSClass::info()->className, operationName);
            } else {
                UNUSED_PARAM(operationName);
                ASSERT(thisObject);
            }

            ASSERT_GC_OBJECT_INHERITS(thisObject, JSClass::info());

            // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject and thisObject.
            return operation(&lexicalGlobalObject, &callFrame, thisObject, WTFMove(promise), WTFMove(promise2));
        });
    }

    // This function is a special case for custom operations want to handle the creation of the promise themselves.
    // It is triggered via the extended attribute [ReturnsOwnPromise].
    template<typename IDLOperation<JSClass>::Operation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::RejectPromise>
    static JSC::EncodedJSValue callReturningOwnPromise(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char* operationName)
    {
        auto* thisObject = IDLOperation<JSClass>::cast(lexicalGlobalObject, callFrame);
        if constexpr (shouldThrow != CastedThisErrorBehavior::Assert) {
            if (UNLIKELY(!thisObject))
                return rejectPromiseWithThisTypeError(lexicalGlobalObject, JSClass::info()->className, operationName);
        } else
            ASSERT(thisObject);

        ASSERT_GC_OBJECT_INHERITS(thisObject, JSClass::info());

        // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject and thisObject.
        return operation(&lexicalGlobalObject, &callFrame, thisObject);
    }

    template<StaticOperation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::RejectPromise>
    static JSC::EncodedJSValue callStatic(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char*)
    {
        return JSC::JSValue::encode(callPromiseFunction(lexicalGlobalObject, callFrame, [] (JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, Ref<DeferredPromise>&& promise) {
            // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject.
            return operation(&lexicalGlobalObject, &callFrame, WTFMove(promise));
        }));
    }

    // This function is a special case for custom operations want to handle the creation of the promise themselves.
    // It is triggered via the extended attribute [ReturnsOwnPromise].
    template<typename IDLOperation<JSClass>::StaticOperation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::RejectPromise>
    static JSC::EncodedJSValue callStaticReturningOwnPromise(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char*)
    {
        // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject.
        return operation(&lexicalGlobalObject, &callFrame);
    }
};

} // namespace WebCore
