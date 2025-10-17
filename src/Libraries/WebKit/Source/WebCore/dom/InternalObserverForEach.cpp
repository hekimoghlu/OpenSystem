/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "InternalObserverForEach.h"

#include "AbortSignal.h"
#include "InternalObserver.h"
#include "JSDOMPromiseDeferred.h"
#include "Observable.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include "VisitorCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverForEach final : public InternalObserver {
public:
    static Ref<InternalObserverForEach> create(ScriptExecutionContext& context, Ref<VisitorCallback>&& callback, Ref<AbortSignal>&& signal, Ref<DeferredPromise>&& promise)
    {
        Ref internalObserver = adoptRef(*new InternalObserverForEach(context, WTFMove(callback), WTFMove(signal), WTFMove(promise)));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

private:
    void next(JSC::JSValue value) final
    {
        auto* globalObject = protectedScriptExecutionContext()->globalObject();
        ASSERT(globalObject);

        Ref vm = globalObject->vm();

        {
            JSC::JSLockHolder lock(vm);

            // The exception is not reported, instead it is forwarded to the
            // abort signal and promise rejection.
            auto scope = DECLARE_CATCH_SCOPE(vm);

            protectedCallback()->handleEventRethrowingException(value, m_idx++);

            JSC::Exception* exception = scope.exception();
            if (UNLIKELY(exception)) {
                scope.clearException();
                auto value = exception->value();
                protectedPromise()->reject<IDLAny>(value);
                Ref { m_signal }->signalAbort(value);
            }
        }
    }

    void error(JSC::JSValue value) final
    {
        protectedPromise()->reject<IDLAny>(value);
    }

    void complete() final
    {
        InternalObserver::complete();
        protectedPromise()->resolve();
    }

    void visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor) const final
    {
        m_callback->visitJSFunction(visitor);
    }

    Ref<DeferredPromise> protectedPromise() const { return m_promise; }
    Ref<VisitorCallback> protectedCallback() const { return m_callback; }

    InternalObserverForEach(ScriptExecutionContext& context, Ref<VisitorCallback>&& callback, Ref<AbortSignal>&& signal, Ref<DeferredPromise>&& promise)
        : InternalObserver(context)
        , m_callback(WTFMove(callback))
        , m_signal(WTFMove(signal))
        , m_promise(WTFMove(promise))
    {
    }

    uint64_t m_idx { 0 };
    const Ref<VisitorCallback> m_callback;
    const Ref<AbortSignal> m_signal;
    const Ref<DeferredPromise> m_promise;
};

void createInternalObserverOperatorForEach(ScriptExecutionContext& context, Observable& observable, Ref<VisitorCallback>&& callback, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    Ref signal = AbortSignal::create(&context);

    Vector<Ref<AbortSignal>> dependentSignals = { signal };
    if (options.signal)
        dependentSignals.append(Ref { *options.signal });
    Ref dependentSignal = AbortSignal::any(context, dependentSignals);

    if (dependentSignal->aborted())
        return promise->reject<IDLAny>(dependentSignal->reason().getValue());

    dependentSignal->addAlgorithm([promise](JSC::JSValue reason) {
        promise->reject<IDLAny>(reason);
    });

    Ref observer = InternalObserverForEach::create(context, WTFMove(callback), WTFMove(signal), WTFMove(promise));

    observable.subscribeInternal(context, WTFMove(observer), SubscribeOptions { .signal = WTFMove(dependentSignal) });
}

} // namespace WebCore
