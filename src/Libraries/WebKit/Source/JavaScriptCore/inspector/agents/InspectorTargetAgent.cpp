/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "InspectorTargetAgent.h"

#include "InspectorTarget.h"
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorTargetAgent);

InspectorTargetAgent::InspectorTargetAgent(FrontendRouter& frontendRouter, BackendDispatcher& backendDispatcher)
    : InspectorAgentBase("Target"_s)
    , m_router(frontendRouter)
    , m_frontendDispatcher(makeUnique<TargetFrontendDispatcher>(frontendRouter))
    , m_backendDispatcher(TargetBackendDispatcher::create(backendDispatcher, this))
{
}

InspectorTargetAgent::~InspectorTargetAgent() = default;

void InspectorTargetAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
    m_isConnected = true;

    connectToTargets();
}

void InspectorTargetAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    disconnectFromTargets();

    m_isConnected = false;
    m_shouldPauseOnStart = false;
}

Protocol::ErrorStringOr<void> InspectorTargetAgent::setPauseOnStart(bool pauseOnStart)
{
    m_shouldPauseOnStart = pauseOnStart;

    return { };
}

Protocol::ErrorStringOr<void> InspectorTargetAgent::resume(const String& targetId)
{
    auto* target = m_targets.get(targetId);
    if (!target)
        return makeUnexpected("Missing target for given targetId"_s);

    if (!target->isPaused())
        return makeUnexpected("Target for given targetId is not paused"_s);

    target->resume();

    return { };
}

Protocol::ErrorStringOr<void> InspectorTargetAgent::sendMessageToTarget(const String& targetId, const String& message)
{
    InspectorTarget* target = m_targets.get(targetId);
    if (!target)
        return makeUnexpected("Missing target for given targetId"_s);

    target->sendMessageToTargetBackend(message);

    return { };
}

void InspectorTargetAgent::sendMessageFromTargetToFrontend(const String& targetId, const String& message)
{
    ASSERT_WITH_MESSAGE(m_targets.get(targetId), "Sending a message from an untracked target to the frontend.");

    m_frontendDispatcher->dispatchMessageFromTarget(targetId, message);
}

static Protocol::Target::TargetInfo::Type targetTypeToProtocolType(InspectorTargetType type)
{
    switch (type) {
    case InspectorTargetType::Page:
        return Protocol::Target::TargetInfo::Type::Page;
    case InspectorTargetType::DedicatedWorker:
        return Protocol::Target::TargetInfo::Type::Worker;
    case InspectorTargetType::ServiceWorker:
        return Protocol::Target::TargetInfo::Type::ServiceWorker;
    }

    ASSERT_NOT_REACHED();
    return Protocol::Target::TargetInfo::Type::Page;
}

static Ref<Protocol::Target::TargetInfo> buildTargetInfoObject(const InspectorTarget& target)
{
    auto result = Protocol::Target::TargetInfo::create()
        .setTargetId(target.identifier())
        .setType(targetTypeToProtocolType(target.type()))
        .release();
    if (target.isProvisional())
        result->setIsProvisional(true);
    if (target.isPaused())
        result->setIsPaused(true);
    return result;
}

void InspectorTargetAgent::targetCreated(InspectorTarget& target)
{
    auto addResult = m_targets.set(target.identifier(), &target);
    ASSERT_UNUSED(addResult, addResult.isNewEntry);

    if (!m_isConnected)
        return;

    if (m_shouldPauseOnStart)
        target.pause();
    target.connect(connectionType());

    m_frontendDispatcher->targetCreated(buildTargetInfoObject(target));
}

void InspectorTargetAgent::targetDestroyed(InspectorTarget& target)
{
    m_targets.remove(target.identifier());

    if (!m_isConnected)
        return;

    m_frontendDispatcher->targetDestroyed(target.identifier());
}

void InspectorTargetAgent::didCommitProvisionalTarget(const String& oldTargetID, const String& committedTargetID)
{
    if (!m_isConnected)
        return;

    auto* target = m_targets.get(committedTargetID);
    if (!target)
        return;

    m_frontendDispatcher->didCommitProvisionalTarget(oldTargetID, committedTargetID);
}

FrontendChannel::ConnectionType InspectorTargetAgent::connectionType() const
{
    return m_router.hasLocalFrontend() ? Inspector::FrontendChannel::ConnectionType::Local : Inspector::FrontendChannel::ConnectionType::Remote;
}

void InspectorTargetAgent::connectToTargets()
{
    for (InspectorTarget* target : m_targets.values()) {
        target->connect(connectionType());
        m_frontendDispatcher->targetCreated(buildTargetInfoObject(*target));
    }
}

void InspectorTargetAgent::disconnectFromTargets()
{
    for (InspectorTarget* target : m_targets.values())
        target->disconnect();
}

} // namespace Inspector
