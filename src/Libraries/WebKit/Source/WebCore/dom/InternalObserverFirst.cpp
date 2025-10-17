/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "InternalObserverFirst.h"

#include "AbortSignal.h"
#include "Exception.h"
#include "ExceptionCode.h"
#include "InternalObserver.h"
#include "JSDOMPromiseDeferred.h"
#include "Observable.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverFirst final : public InternalObserver {
public:
    static Ref<InternalObserverFirst> create(ScriptExecutionContext& context, Ref<AbortSignal>&& signal, Ref<DeferredPromise>&& promise)
    {
        Ref internalObserver = adoptRef(*new InternalObserverFirst(context, WTFMove(signal), WTFMove(promise)));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

private:
    void next(JSC::JSValue value) final
    {
        protectedPromise()->resolve<IDLAny>(value);
        Ref { m_signal }->signalAbort(JSC::jsUndefined());
    }

    void error(JSC::JSValue value) final
    {
        protectedPromise()->reject<IDLAny>(value);
    }

    void complete() final
    {
        InternalObserver::complete();
        protectedPromise()->reject(Exception { ExceptionCode::RangeError, "No values in Observable"_s });
    }

    void visitAdditionalChildren(JSC::AbstractSlotVisitor&) const final
    {
    }

    Ref<DeferredPromise> protectedPromise() const { return m_promise; }

    InternalObserverFirst(ScriptExecutionContext& context, Ref<AbortSignal>&& signal, Ref<DeferredPromise>&& promise)
        : InternalObserver(context)
        , m_signal(WTFMove(signal))
        , m_promise(WTFMove(promise))
    {
    }

    const Ref<AbortSignal> m_signal;
    const Ref<DeferredPromise> m_promise;
};

void createInternalObserverOperatorFirst(ScriptExecutionContext& context, Observable& observable, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
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

    Ref observer = InternalObserverFirst::create(context, WTFMove(signal), WTFMove(promise));

    observable.subscribeInternal(context, WTFMove(observer), SubscribeOptions { .signal = WTFMove(dependentSignal) });
}

} // namespace WebCore
