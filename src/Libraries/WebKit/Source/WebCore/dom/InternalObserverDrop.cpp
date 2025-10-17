/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#include "InternalObserverDrop.h"

#include "InternalObserver.h"
#include "Observable.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverDrop final : public InternalObserver {
public:
    static Ref<InternalObserverDrop> create(ScriptExecutionContext& context, Ref<Subscriber> subscriber, uint64_t amount)
    {
        Ref internalObserver = adoptRef(*new InternalObserverDrop(context, subscriber, amount));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

    class SubscriberCallbackDrop final : public SubscriberCallback {
    public:
        static Ref<SubscriberCallbackDrop> create(ScriptExecutionContext& context, Ref<Observable> source, uint64_t amount)
        {
            return adoptRef(*new InternalObserverDrop::SubscriberCallbackDrop(context, source, amount));
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
            m_sourceObservable->subscribeInternal(*context, InternalObserverDrop::create(*context, subscriber, m_amount), options);

            return { };
        }

        CallbackResult<void> handleEventRethrowingException(Subscriber& subscriber) final
        {
            return handleEvent(subscriber);
        }

    private:
        SubscriberCallbackDrop(ScriptExecutionContext& context, Ref<Observable> source, uint64_t amount)
            : SubscriberCallback(&context)
            , m_sourceObservable(source)
            , m_amount(amount)
        { }

        bool hasCallback() const final { return true; }

        const Ref<Observable> m_sourceObservable;
        uint64_t m_amount;
    };

private:
    void next(JSC::JSValue value) final
    {
        if (!m_amount) {
            protectedSubscriber()->next(value);
            return;
        }

        m_amount -= 1;
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
    }

    Ref<Subscriber> protectedSubscriber() const { return m_subscriber; }

    InternalObserverDrop(ScriptExecutionContext& context, Ref<Subscriber> subscriber, uint64_t amount)
        : InternalObserver(context)
        , m_subscriber(subscriber)
        , m_amount(amount)
    { }

    const Ref<Subscriber> m_subscriber;
    uint64_t m_amount;
};

Ref<SubscriberCallback> createSubscriberCallbackDrop(ScriptExecutionContext& context, Ref<Observable> observable, uint64_t amount)
{
    return InternalObserverDrop::SubscriberCallbackDrop::create(context, observable, amount);
}

} // namespace WebCore
