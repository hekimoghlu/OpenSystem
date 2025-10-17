/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include "InternalObserverInspect.h"

#include "InternalObserver.h"
#include "JSSubscriptionObserverCallback.h"
#include "Observable.h"
#include "ObservableInspector.h"
#include "ScriptExecutionContext.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

class InternalObserverInspect final : public InternalObserver {
public:
    static Ref<InternalObserverInspect> create(ScriptExecutionContext& context, Ref<Subscriber>&& subscriber, ObservableInspector&& inspector)
    {
        Ref internalObserver = adoptRef(*new InternalObserverInspect(context, WTFMove(subscriber), WTFMove(inspector)));
        internalObserver->suspendIfNeeded();
        return internalObserver;
    }

    class SubscriberCallbackInspect final : public SubscriberCallback {
    public:
        static Ref<SubscriberCallbackInspect> create(ScriptExecutionContext& context, Ref<Observable>&& source, ObservableInspector&& inspector)
        {
            return adoptRef(*new SubscriberCallbackInspect(context, WTFMove(source), WTFMove(inspector)));
        }

        CallbackResult<void> handleEvent(Subscriber& subscriber) final
        {
            RefPtr context = scriptExecutionContext();

            if (!context) {
                subscriber.complete();
                return { };
            }

            if (RefPtr subscribe = m_inspector.subscribe) {
                auto* globalObject = protectedScriptExecutionContext()->globalObject();
                ASSERT(globalObject);

                Ref vm = globalObject->vm();

                JSC::JSLockHolder lock(vm);
                auto scope = DECLARE_CATCH_SCOPE(vm);

                subscribe->handleEventRethrowingException();

                JSC::Exception* exception = scope.exception();
                if (UNLIKELY(exception)) {
                    scope.clearException();
                    subscriber.error(exception->value());
                    return { };
                }
            }

            Ref inspect = InternalObserverInspect::create(*context, subscriber, ObservableInspector { m_inspector });
            Ref { m_sourceObservable }->subscribeInternal(*context, WTFMove(inspect), SubscribeOptions { &subscriber.signal() });

            return { };
        }

        CallbackResult<void> handleEventRethrowingException(Subscriber& subscriber) final
        {
            return handleEvent(subscriber);
        }

    private:
        bool hasCallback() const final { return true; }

        SubscriberCallbackInspect(ScriptExecutionContext& context, Ref<Observable>&& source, ObservableInspector&& inspector)
            : SubscriberCallback(&context)
            , m_sourceObservable(WTFMove(source))
            , m_inspector(WTFMove(inspector))
        { }

        const Ref<Observable> m_sourceObservable;
        const ObservableInspector m_inspector;
    };

private:
    void next(JSC::JSValue value) final
    {
        if (RefPtr next = m_inspector.next) {
            Ref vm = this->vm();
            JSC::JSLockHolder lock(vm);
            auto scope = DECLARE_CATCH_SCOPE(vm);

            next->handleEventRethrowingException(value);

            JSC::Exception* exception = scope.exception();
            if (UNLIKELY(exception)) {
                scope.clearException();
                protectedSubscriber()->error(exception->value());
                return;
            }
        }

        protectedSubscriber()->next(value);
    }

    void error(JSC::JSValue value) final
    {
        removeAbortHandler();

        if (RefPtr error = m_inspector.error) {
            Ref vm = this->vm();
            JSC::JSLockHolder lock(vm);
            auto scope = DECLARE_CATCH_SCOPE(vm);

            error->handleEventRethrowingException(value);

            JSC::Exception* exception = scope.exception();
            if (UNLIKELY(exception)) {
                scope.clearException();
                protectedSubscriber()->error(exception->value());
                return;
            }
        }

        protectedSubscriber()->error(value);
    }

    void complete() final
    {
        InternalObserver::complete();

        removeAbortHandler();

        if (RefPtr complete = m_inspector.complete) {
            Ref vm = this->vm();
            JSC::JSLockHolder lock(vm);
            auto scope = DECLARE_CATCH_SCOPE(vm);

            complete->handleEventRethrowingException();

            JSC::Exception* exception = scope.exception();
            if (UNLIKELY(exception)) {
                scope.clearException();
                protectedSubscriber()->error(exception->value());
                return;
            }
        }

        protectedSubscriber()->complete();
    }

    void visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor) const final
    {
        m_subscriber->visitAdditionalChildren(visitor);
        if (m_inspector.next)
            SUPPRESS_UNCOUNTED_ARG m_inspector.next->visitJSFunction(visitor);
        if (m_inspector.error)
            SUPPRESS_UNCOUNTED_ARG m_inspector.error->visitJSFunction(visitor);
        if (m_inspector.complete)
            SUPPRESS_UNCOUNTED_ARG m_inspector.complete->visitJSFunction(visitor);
        if (m_inspector.subscribe)
            SUPPRESS_UNCOUNTED_ARG m_inspector.subscribe->visitJSFunction(visitor);
        if (m_inspector.abort)
            SUPPRESS_UNCOUNTED_ARG m_inspector.abort->visitJSFunction(visitor);
    }

    void removeAbortHandler()
    {
        if (!m_abortAlgorithmHandler)
            return;

        auto handle = std::exchange(m_abortAlgorithmHandler, std::nullopt);
        protectedSubscriber()->protectedSignal()->removeAlgorithm(*handle);
    }

    JSC::VM& vm() const
    {
        auto* globalObject = protectedScriptExecutionContext()->globalObject();
        ASSERT(globalObject);
        return globalObject->vm();
    }

    Ref<Subscriber> protectedSubscriber() const
    {
        return m_subscriber;
    }

    InternalObserverInspect(ScriptExecutionContext& context, Ref<Subscriber>&& subscriber, ObservableInspector&& inspector)
        : InternalObserver(context)
        , m_subscriber(WTFMove(subscriber))
        , m_inspector(WTFMove(inspector))
    {
        if (RefPtr abort = m_inspector.abort) {
            Ref signal = protectedSubscriber()->signal();
            m_abortAlgorithmHandler = signal->addAlgorithm([abort = WTFMove(abort)](JSC::JSValue reason) {
                abort->handleEvent(reason);
            });
        }
    }

    const Ref<Subscriber> m_subscriber;
    const ObservableInspector m_inspector;
    std::optional<uint32_t> m_abortAlgorithmHandler;
};

Ref<SubscriberCallback> createSubscriberCallbackInspect(ScriptExecutionContext& context, Ref<Observable>&& observable, RefPtr<JSSubscriptionObserverCallback>&& next)
{
    return InternalObserverInspect::SubscriberCallbackInspect::create(context, WTFMove(observable), ObservableInspector { .next = WTFMove(next) });
}

Ref<SubscriberCallback> createSubscriberCallbackInspect(ScriptExecutionContext& context, Ref<Observable>&& observable, ObservableInspector&& inspector)
{
    return InternalObserverInspect::SubscriberCallbackInspect::create(context, WTFMove(observable), WTFMove(inspector));
}

} // namespace WebCore
