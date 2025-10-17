/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "PermissionStatus.h"

#include "ClientOrigin.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "EventNames.h"
#include "MainThreadPermissionObserver.h"
#include "PermissionController.h"
#include "PermissionState.h"
#include "Permissions.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <wtf/HashMap.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PermissionStatus);

static HashMap<MainThreadPermissionObserverIdentifier, std::unique_ptr<MainThreadPermissionObserver>>& allMainThreadPermissionObservers()
{
    static MainThreadNeverDestroyed<HashMap<MainThreadPermissionObserverIdentifier, std::unique_ptr<MainThreadPermissionObserver>>> map;
    return map;
}

Ref<PermissionStatus> PermissionStatus::create(ScriptExecutionContext& context, PermissionState state, PermissionDescriptor descriptor, PermissionQuerySource source, WeakPtr<Page>&& page)
{
    auto status = adoptRef(*new PermissionStatus(context, state, descriptor, source, WTFMove(page)));
    status->suspendIfNeeded();
    return status;
}

PermissionStatus::PermissionStatus(ScriptExecutionContext& context, PermissionState state, PermissionDescriptor descriptor, PermissionQuerySource source, WeakPtr<Page>&& page)
    : ActiveDOMObject(&context)
    , m_state(state)
    , m_descriptor(descriptor)
    , m_mainThreadPermissionObserverIdentifier(MainThreadPermissionObserverIdentifier::generate())
{
    RefPtr origin = context.securityOrigin();
    auto originData = origin ? origin->data() : SecurityOriginData { };
    ClientOrigin clientOrigin { context.topOrigin().data(), WTFMove(originData) };

    ensureOnMainThread([weakThis = ThreadSafeWeakPtr { *this }, contextIdentifier = context.identifier(), state = m_state, descriptor = m_descriptor, source, page = WTFMove(page), origin = WTFMove(clientOrigin).isolatedCopy(), identifier = m_mainThreadPermissionObserverIdentifier]() mutable {
        auto mainThreadPermissionObserver = makeUnique<MainThreadPermissionObserver>(WTFMove(weakThis), contextIdentifier, state, descriptor, source, WTFMove(page), WTFMove(origin));
        allMainThreadPermissionObservers().add(identifier, WTFMove(mainThreadPermissionObserver));
    });
}

PermissionStatus::~PermissionStatus()
{
    callOnMainThread([identifier = m_mainThreadPermissionObserverIdentifier] {
        allMainThreadPermissionObservers().remove(identifier);
    });
}

void PermissionStatus::stateChanged(PermissionState newState)
{
    if (m_state == newState)
        return;

    RefPtr context = scriptExecutionContext();
    if (!context)
        return;

    RefPtr document = dynamicDowncast<Document>(context.get());
    if (document && !document->isFullyActive())
        return;

    m_state = newState;
    queueTaskToDispatchEvent(*this, TaskSource::Permission, Event::create(eventNames().changeEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

bool PermissionStatus::virtualHasPendingActivity() const
{
    if (!m_hasChangeEventListener)
        return false;

    if (auto* document = dynamicDowncast<Document>(scriptExecutionContext()))
        return document->hasBrowsingContext();

    return true;
}

void PermissionStatus::eventListenersDidChange()
{
    m_hasChangeEventListener = hasEventListeners(eventNames().changeEvent);
}

} // namespace WebCore
