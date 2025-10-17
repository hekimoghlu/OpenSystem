/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#include "InternalObserverFromScript.h"

#include "JSSubscriptionObserverCallback.h"
#include "ScriptExecutionContext.h"
#include "SubscriptionObserver.h"

namespace WebCore {

Ref<InternalObserverFromScript> InternalObserverFromScript::create(ScriptExecutionContext& context, RefPtr<JSSubscriptionObserverCallback> callback)
{
    Ref internalObserver = adoptRef(*new InternalObserverFromScript(context, callback));
    internalObserver->suspendIfNeeded();
    return internalObserver;
}

Ref<InternalObserverFromScript> InternalObserverFromScript::create(ScriptExecutionContext& context, SubscriptionObserver& subscription)
{
    Ref internalObserver = adoptRef(*new InternalObserverFromScript(context, subscription));
    internalObserver->suspendIfNeeded();
    return internalObserver;
}

void InternalObserverFromScript::next(JSC::JSValue value)
{
    if (RefPtr next = m_next)
        next->handleEvent(value);
}

void InternalObserverFromScript::error(JSC::JSValue value)
{
    if (RefPtr error = m_error) {
        error->handleEvent(value);
        return;
    }

    InternalObserver::error(value);
}

void InternalObserverFromScript::complete()
{
    if (RefPtr complete = m_complete)
        complete->handleEvent();

    m_active = false;
}

void InternalObserverFromScript::visitAdditionalChildren(JSC::AbstractSlotVisitor& visitor) const
{
    if (RefPtr next = m_next)
        next->visitJSFunction(visitor);

    if (RefPtr error = m_error)
        error->visitJSFunction(visitor);

    if (RefPtr complete = m_complete)
        complete->visitJSFunction(visitor);
}

InternalObserverFromScript::InternalObserverFromScript(ScriptExecutionContext& context, RefPtr<JSSubscriptionObserverCallback> callback)
    : InternalObserver(context)
    , m_next(callback)
    , m_error(nullptr)
    , m_complete(nullptr) { }

InternalObserverFromScript::InternalObserverFromScript(ScriptExecutionContext& context, SubscriptionObserver& subscription)
    : InternalObserver(context)
    , m_next(subscription.next)
    , m_error(subscription.error)
    , m_complete(subscription.complete) { }

} // namespace WebCore
