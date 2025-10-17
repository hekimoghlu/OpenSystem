/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "InternalObserverMap.h"

#include "InternalObserver.h"
#include "MapperCallback.h"
#include "Observable.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverMap final : public InternalObserver {
public:
    static Ref<InternalObserverMap> create(ScriptExecutionContext& context, Ref<Subscriber> subscriber, Ref<MapperCallback> mapper)
    {
        Ref internalObserver = adoptRef(*new InternalObserverMap(context, subscriber, mapper));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

    class SubscriberCallbackMap final : public SubscriberCallback {
    public:
        static Ref<SubscriberCallbackMap> create(ScriptExecutionContext& context, Ref<Observable> source, Ref<MapperCallback> mapper)
        {
            return adoptRef(*new InternalObserverMap::SubscriberCallbackMap(context, source, mapper));
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
            m_sourceObservable->subscribeInternal(*context, InternalObserverMap::create(*context, subscriber, m_mapper), options);

            return { };
        }

        CallbackResult<void> handleEventRethrowingException(Subscriber& subscriber) final
        {
            return handleEvent(subscriber);
        }

    private:
        SubscriberCallbackMap(ScriptExecutionContext& context, Ref<Observable> source, Ref<MapperCallback> mapper)
            : SubscriberCallback(&context)
            , m_sourceObservable(source)
            , m_mapper(mapper)
        { }

        bool hasCallback() const final { return true; }

        const Ref<Observable> m_sourceObservable;
        const Ref<MapperCallback> m_mapper;
    };

private:
    void next(JSC::JSValue value) final
    {
        RefPtr context = scriptExecutionContext();
        if (!context)
            return;

        Ref vm = context->globalObject()->vm();
        JSC::JSLockHolder lock(vm);

        // The exception is not reported, instead it is forwarded to the
        // error handler.
        JSC::Exception* previousException = nullptr;
        {
            auto catchScope = DECLARE_CATCH_SCOPE(vm);
            auto result = protectedMapper()->handleEventRethrowingException(value, m_idx);
            previousException = catchScope.exception();
            if (previousException) {
                catchScope.clearException();
                protectedSubscriber()->error(previousException->value());
                return;
            }

            m_idx += 1;

            if (result.type() == CallbackResultType::Success)
                protectedSubscriber()->next(result.releaseReturnValue());
        }
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
        m_mapper->visitJSFunction(visitor);
    }

    Ref<Subscriber> protectedSubscriber() const { return m_subscriber; }
    Ref<MapperCallback> protectedMapper() const { return m_mapper; }

    InternalObserverMap(ScriptExecutionContext& context, Ref<Subscriber> subscriber, Ref<MapperCallback> mapper)
        : InternalObserver(context)
        , m_subscriber(subscriber)
        , m_mapper(mapper)
    { }

    const Ref<Subscriber> m_subscriber;
    const Ref<MapperCallback> m_mapper;
    uint64_t m_idx { 0 };
};

Ref<SubscriberCallback> createSubscriberCallbackMap(ScriptExecutionContext& context, Ref<Observable> observable, Ref<MapperCallback> mapper)
{
    return InternalObserverMap::SubscriberCallbackMap::create(context, observable, mapper);
}

} // namespace WebCore
