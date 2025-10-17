/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#include "MainThreadPermissionObserver.h"

#include "ClientOrigin.h"
#include "Document.h"
#include "PermissionController.h"
#include "PermissionState.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MainThreadPermissionObserver);

MainThreadPermissionObserver::MainThreadPermissionObserver(ThreadSafeWeakPtr<PermissionStatus>&& permissionStatus, ScriptExecutionContextIdentifier contextIdentifier, PermissionState state, PermissionDescriptor descriptor, PermissionQuerySource source, WeakPtr<Page>&& page, ClientOrigin&& origin)
    : m_permissionStatus(WTFMove(permissionStatus))
    , m_contextIdentifier(contextIdentifier)
    , m_state(state)
    , m_descriptor(descriptor)
    , m_source(source)
    , m_page(WTFMove(page))
    , m_origin(WTFMove(origin))
{
    ASSERT(isMainThread());
    PermissionController::shared().addObserver(*this);
}

MainThreadPermissionObserver::~MainThreadPermissionObserver()
{
    ASSERT(isMainThread());
    PermissionController::shared().removeObserver(*this);
}

void MainThreadPermissionObserver::stateChanged(PermissionState newPermissionState)
{
    m_state = newPermissionState;

    ScriptExecutionContext::ensureOnContextThread(m_contextIdentifier, [weakPermissionStatus = m_permissionStatus, newPermissionState](auto&) {
        if (RefPtr permissionStatus = weakPermissionStatus.get())
            permissionStatus->stateChanged(newPermissionState);
    });
}

}
