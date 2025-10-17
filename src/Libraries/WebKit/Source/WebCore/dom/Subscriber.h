/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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

#include "AbortController.h"
#include "ActiveDOMObject.h"
#include "InternalObserver.h"
#include "ScriptWrappable.h"
#include "SubscribeOptions.h"
#include "VoidCallback.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class ScriptExecutionContext;

class Subscriber final : public ActiveDOMObject, public ScriptWrappable, public RefCounted<Subscriber> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Subscriber);

public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void next(JSC::JSValue);
    void complete();
    void error(JSC::JSValue);
    void addTeardown(Ref<VoidCallback>);

    bool active() { return m_active; }
    AbortSignal& signal() { return m_signal.get(); }

    Ref<AbortSignal> protectedSignal() const { return m_signal; }

    static Ref<Subscriber> create(ScriptExecutionContext&, Ref<InternalObserver>&&, const SubscribeOptions&);

    void reportErrorObject(JSC::JSValue);

    // JSCustomMarkFunction; for JSSubscriberCustom
    Vector<VoidCallback*> teardownCallbacksConcurrently();
    InternalObserver* observerConcurrently();
    void visitAdditionalChildren(JSC::AbstractSlotVisitor&);

private:
    explicit Subscriber(ScriptExecutionContext&, Ref<InternalObserver>&&, const SubscribeOptions&);

    void followSignal(AbortSignal&);
    void close(JSC::JSValue);

    bool isActive() const
    {
        return m_active && !isInactiveDocument();
    }

    bool isInactiveDocument() const;

    // ActiveDOMObject
    void stop() final
    {
        Locker locker { m_teardownsLock };
        m_teardowns.clear();
    }
    bool virtualHasPendingActivity() const final { return m_active; }

    bool m_active = true;
    Lock m_teardownsLock;
    Ref<AbortSignal> m_signal;
    Ref<InternalObserver> m_observer;
    SubscribeOptions m_options;
    Vector<Ref<VoidCallback>> m_teardowns WTF_GUARDED_BY_LOCK(m_teardownsLock);
};

} // namespace WebCore
