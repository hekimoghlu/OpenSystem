/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#include "InternalObserverLast.h"

#include "AbortController.h"
#include "AbortSignal.h"
#include "Exception.h"
#include "ExceptionCode.h"
#include "InternalObserver.h"
#include "JSDOMPromiseDeferred.h"
#include "JSValueInWrappedObject.h"
#include "Observable.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverLast final : public InternalObserver {
public:
    static Ref<InternalObserverLast> create(ScriptExecutionContext& context, Ref<DeferredPromise>&& promise)
    {
        Ref internalObserver = adoptRef(*new InternalObserverLast(context, WTFMove(promise)));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

private:
    void next(JSC::JSValue value) final
    {
        m_lastValue.setWeakly(value);
    }

    void error(JSC::JSValue value) final
    {
        protectedPromise()->reject<IDLAny>(value);
    }

    void complete() final
    {
        InternalObserver::complete();

        if (UNLIKELY(!m_lastValue))
            return protectedPromise()->reject(Exception { ExceptionCode::RangeError, "No values in Observable"_s });

        protectedPromise()->resolve<IDLAny>(m_lastValue.getValue());
    }

    void visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor) const final
    {
        m_lastValue.visit(visitor);
    }

    Ref<DeferredPromise> protectedPromise() const { return m_promise; }

    InternalObserverLast(ScriptExecutionContext& context, Ref<DeferredPromise>&& promise)
        : InternalObserver(context)
        , m_promise(WTFMove(promise))
    {
    }

    JSValueInWrappedObject m_lastValue;
    const Ref<DeferredPromise> m_promise;
};

void createInternalObserverOperatorLast(ScriptExecutionContext& context, Observable& observable, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    if (RefPtr signal = options.signal) {
        if (signal->aborted())
            return promise->reject<IDLAny>(signal->reason().getValue());

        signal->addAlgorithm([promise](JSC::JSValue reason) {
            promise->reject<IDLAny>(reason);
        });
    }

    Ref observer = InternalObserverLast::create(context, WTFMove(promise));

    observable.subscribeInternal(context, WTFMove(observer), options);
}

} // namespace WebCore
