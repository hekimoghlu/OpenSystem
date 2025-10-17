/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include "ClientOrigin.h"
#include "EventTarget.h"
#include "MainThreadPermissionObserverIdentifier.h"
#include "Page.h"
#include "PermissionDescriptor.h"
#include "PermissionName.h"
#include "PermissionQuerySource.h"
#include "PermissionState.h"

namespace WebCore {

class ScriptExecutionContext;

class PermissionStatus final : public ActiveDOMObject, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<PermissionStatus>, public EventTarget  {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PermissionStatus);
public:
    static Ref<PermissionStatus> create(ScriptExecutionContext&, PermissionState, PermissionDescriptor, PermissionQuerySource, WeakPtr<Page>&&);
    ~PermissionStatus();

    PermissionState state() const { return m_state; }
    PermissionName name() const { return m_descriptor.name; }

    void stateChanged(PermissionState);

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

private:
    PermissionStatus(ScriptExecutionContext&, PermissionState, PermissionDescriptor, PermissionQuerySource, WeakPtr<Page>&&);

    // ActiveDOMObject
    bool virtualHasPendingActivity() const final;

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::PermissionStatus; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    PermissionState m_state;
    PermissionDescriptor m_descriptor;
    MainThreadPermissionObserverIdentifier m_mainThreadPermissionObserverIdentifier;
    bool m_hasChangeEventListener { false };
};

} // namespace WebCore
