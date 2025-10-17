/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include "ContextDestructionObserverInlines.h"
#include "EventTarget.h"
#include "ScreenOrientationLockType.h"
#include "ScreenOrientationManager.h"
#include "ScreenOrientationType.h"
#include "VisibilityChangeClient.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class DeferredPromise;

class ScreenOrientation final : public ActiveDOMObject, public EventTarget, public ScreenOrientationManagerObserver, public VisibilityChangeClient, public RefCounted<ScreenOrientation> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ScreenOrientation);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<ScreenOrientation> create(Document*);
    ~ScreenOrientation();

    USING_CAN_MAKE_WEAKPTR(ScreenOrientationManagerObserver);

    using LockType = ScreenOrientationLockType;
    using Type = ScreenOrientationType;

    void lock(LockType, Ref<DeferredPromise>&&);
    ExceptionOr<void> unlock();
    Type type() const;
    uint16_t angle() const;

private:
    ScreenOrientation(Document*);

    Document* document() const;
    ScreenOrientationManager* manager() const;

    bool shouldListenForChangeNotification() const;

    // VisibilityChangeClient
    void visibilityStateChanged() final;

    // ScreenOrientationManagerObserver
    void screenOrientationDidChange(ScreenOrientationType) final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::ScreenOrientation; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { RefCounted::ref(); }
    void derefEventTarget() final { RefCounted::deref(); }
    void eventListenersDidChange() final;

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;
    void suspend(ReasonForSuspension) final;
    void resume() final;
    void stop() final;

    bool m_hasChangeEventListener { false };
};

} // namespace WebCore
