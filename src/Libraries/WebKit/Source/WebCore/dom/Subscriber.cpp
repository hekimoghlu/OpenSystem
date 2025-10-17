/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#include "Subscriber.h"

#include "AbortSignal.h"
#include "Document.h"
#include "InternalObserver.h"
#include "JSDOMExceptionHandling.h"
#include "SubscriberCallback.h"
#include "SubscriptionObserverCallback.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

Ref<Subscriber> Subscriber::create(ScriptExecutionContext& context, Ref<InternalObserver>&& observer, const SubscribeOptions& options)
{
    return adoptRef(*new Subscriber(context, WTFMove(observer), options));
}

Subscriber::Subscriber(ScriptExecutionContext& context, Ref<InternalObserver>&& observer, const SubscribeOptions& options)
    : ActiveDOMObject(&context)
    , m_signal(AbortSignal::create(&context))
    , m_observer(observer)
    , m_options(options)
{
    followSignal(protectedSignal());
    if (RefPtr signal = options.signal)
        followSignal(*signal);
    suspendIfNeeded();
}

void Subscriber::next(JSC::JSValue value)
{
    if (!isActive())
        return;

    m_observer->next(value);
}

void Subscriber::error(JSC::JSValue error)
{
    if (!m_active) {
        reportErrorObject(error);
        return;
    }

    if (isInactiveDocument())
        return;

    close(error);

    m_observer->error(error);
}

void Subscriber::complete()
{
    if (!isActive())
        return;

    close(JSC::jsUndefined());

    m_observer->complete();
}

void Subscriber::addTeardown(Ref<VoidCallback> callback)
{
    if (isInactiveDocument())
        return;

    if (m_active) {
        Locker locker { m_teardownsLock };
        m_teardowns.append(callback);
    } else
        callback->handleEvent();
}

void Subscriber::followSignal(AbortSignal& signal)
{
    if (signal.aborted())
        close(signal.reason().getValue());
    else {
        signal.addAlgorithm([this](JSC::JSValue reason) {
            close(reason);
        });
    }
}

void Subscriber::close(JSC::JSValue reason)
{
    if (!m_active || !scriptExecutionContext())
        return;

    m_active = false;

    protectedSignal()->signalAbort(reason);

    {
        Locker locker { m_teardownsLock };
        for (auto teardown = m_teardowns.rbegin(); teardown != m_teardowns.rend(); ++teardown) {
            if (isInactiveDocument())
                return;
            (*teardown)->handleEvent();
        }
    }

    stop();
}

bool Subscriber::isInactiveDocument() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    return (document && !document->isFullyActive());
}

void Subscriber::reportErrorObject(JSC::JSValue value)
{
    auto* context = scriptExecutionContext();
    if (!context)
        return;

    auto* globalObject = context->globalObject();
    if (!globalObject)
        return;

    Ref vm = globalObject->vm();
    JSC::JSLockHolder lock(vm);

    reportException(globalObject, JSC::Exception::create(vm, value));
}

Vector<VoidCallback*> Subscriber::teardownCallbacksConcurrently()
{
    Locker locker { m_teardownsLock };
    return m_teardowns.map([](auto& callback) {
        return callback.ptr();
    });
}

InternalObserver* Subscriber::observerConcurrently()
{
    return &m_observer.get();
}

void Subscriber::visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor)
{
    for (auto* teardown : teardownCallbacksConcurrently())
        teardown->visitJSFunction(visitor);

    observerConcurrently()->visitAdditionalChildren(visitor);
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Subscriber);

} // namespace WebCore
