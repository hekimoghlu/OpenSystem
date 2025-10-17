/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "WakeLockType.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class DeferredPromise;
class WakeLockManager;

class WakeLockSentinel final : public RefCounted<WakeLockSentinel>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WakeLockSentinel);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<WakeLockSentinel> create(Document& document, WakeLockType type)
    {
        auto sentinel = adoptRef(*new WakeLockSentinel(document, type));
        sentinel->suspendIfNeeded();
        return sentinel;
    }

    bool released() const { return m_wasReleased; }
    WakeLockType type() const { return m_type; }
    void release(Ref<DeferredPromise>&&);
    void release(WakeLockManager&);

private:
    WakeLockSentinel(Document&, WakeLockType);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WakeLockSentinel; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { RefCounted::ref(); }
    void derefEventTarget() final { RefCounted::deref(); }
    void eventListenersDidChange() final;

    WakeLockType m_type;
    bool m_wasReleased { false };
    bool m_hasReleaseEventListener { false };
};

} // namespace WebCore
