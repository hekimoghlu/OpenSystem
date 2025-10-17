/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#include "InternalObserverFilter.h"

#include "InternalObserver.h"
#include "Observable.h"
#include "PredicateCallback.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverFilter final : public InternalObserver {
public:
    static Ref<InternalObserverFilter> create(ScriptExecutionContext& context, Ref<Subscriber> subscriber, Ref<PredicateCallback> predicate)
    {
        Ref internalObserver = adoptRef(*new InternalObserverFilter(context, subscriber, predicate));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

    class SubscriberCallbackFilter final : public SubscriberCallback {
    public:
        static Ref<SubscriberCallbackFilter> create(ScriptExecutionContext& context, Ref<Observable> source, Ref<PredicateCallback> predicate)
        {
            return adoptRef(*new InternalObserverFilter::SubscriberCallbackFilter(context, source, predicate));
        }

        CallbackResult<void> handleEvent(Subscriber& subscriber) final
        {
            RefPtr context = scriptExecutionContext();

            if (!context) {
                subscriber.complete();
                return { };
            }

            SubscribeOptions options;
            options.signal = &subscriber.signal();
            m_sourceObservable->subscribeInternal(*context, InternalObserverFilter::create(*context, subscriber, m_predicate), options);

            return { };
        }

        CallbackResult<void> handleEventRethrowingException(Subscriber& subscriber) final
        {
            return handleEvent(subscriber);
        }

    private:
        SubscriberCallbackFilter(ScriptExecutionContext& context, Ref<Observable> source, Ref<PredicateCallback> predicate)
            : SubscriberCallback(&context)
            , m_sourceObservable(source)
            , m_predicate(predicate)
        { }

        bool hasCallback() const final { return true; }

        const Ref<Observable> m_sourceObservable;
        const Ref<PredicateCallback> m_predicate;
    };

private:
    void next(JSC::JSValue value) final
    {
        RefPtr context = scriptExecutionContext();
        if (!context)
            return;

        Ref vm = context->globalObject()->vm();
        JSC::JSLockHolder lock(vm);

        auto matches = false;

        // The exception is not reported, instead it is forwarded to the
        // error handler.
        JSC::Exception* previousException = nullptr;
        {
            auto catchScope = DECLARE_CATCH_SCOPE(vm);
            auto result = protectedPredicate()->handleEventRethrowingException(value, m_idx);
            previousException = catchScope.exception();
            if (previousException) {
                catchScope.clearException();
                protectedSubscriber()->error(previousException->value());
                return;
            }

            if (result.type() == CallbackResultType::Success)
                matches = result.releaseReturnValue();
        }

        m_idx += 1;

        if (matches)
            protectedSubscriber()->next(value);
    }

    void error(JSC::JSValue value) final
    {
        protectedSubscriber()->error(value);
    }

    void complete() final
    {
        InternalObserver::complete();
        protectedSubscriber()->complete();
    }

    void visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor) const final
    {
        m_subscriber->visitAdditionalChildren(visitor);
        m_predicate->visitJSFunction(visitor);
    }

    Ref<Subscriber> protectedSubscriber() const { return m_subscriber; }
    Ref<PredicateCallback> protectedPredicate() const { return m_predicate; }

    InternalObserverFilter(ScriptExecutionContext& context, Ref<Subscriber> subscriber, Ref<PredicateCallback> predicate)
        : InternalObserver(context)
        , m_subscriber(subscriber)
        , m_predicate(predicate)
    { }

    const Ref<Subscriber> m_subscriber;
    const Ref<PredicateCallback> m_predicate;
    uint64_t m_idx { 0 };
};

Ref<SubscriberCallback> createSubscriberCallbackFilter(ScriptExecutionContext& context, Ref<Observable> observable, Ref<PredicateCallback> predicate)
{
    return InternalObserverFilter::SubscriberCallbackFilter::create(context, observable, predicate);
}

} // namespace WebCore
