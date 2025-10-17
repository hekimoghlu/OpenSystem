/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include "ClientOrigin.h"
#include "MainThreadPermissionObserverIdentifier.h"
#include "PermissionDescriptor.h"
#include "PermissionObserver.h"
#include "PermissionQuerySource.h"
#include "PermissionState.h"
#include "PermissionStatus.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Page;

class MainThreadPermissionObserver final : public PermissionObserver {
    WTF_MAKE_NONCOPYABLE(MainThreadPermissionObserver);
    WTF_MAKE_TZONE_ALLOCATED(MainThreadPermissionObserver);
public:
    MainThreadPermissionObserver(ThreadSafeWeakPtr<PermissionStatus>&&, ScriptExecutionContextIdentifier, PermissionState, PermissionDescriptor, PermissionQuerySource, WeakPtr<Page>&&, ClientOrigin&&);
    ~MainThreadPermissionObserver();

private:
    // PermissionObserver
    PermissionState currentState() const final { return m_state; }
    void stateChanged(PermissionState) final;
    const ClientOrigin& origin() const final { return m_origin; }
    PermissionDescriptor descriptor() const final { return m_descriptor; }
    PermissionQuerySource source() const final { return m_source; }
    const WeakPtr<Page>& page() const final { return m_page; }

    ThreadSafeWeakPtr<PermissionStatus> m_permissionStatus;
    ScriptExecutionContextIdentifier m_contextIdentifier;
    PermissionState m_state;
    PermissionDescriptor m_descriptor;
    PermissionQuerySource m_source;
    WeakPtr<Page> m_page;
    ClientOrigin m_origin;
};

} // namespace WebCore
