/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include "RemoteInspector.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteAutomationTarget.h"
#include "RemoteConnectionToTarget.h"
#include "RemoteInspectionTarget.h"
#include "RemoteInspectorConstants.h"
#include <wtf/MainThread.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

bool RemoteInspector::startEnabled = true;
#if PLATFORM(COCOA)
std::atomic<bool> RemoteInspector::needMachSandboxExtension = false;
#endif

#if !PLATFORM(COCOA)
RemoteInspector::~RemoteInspector() = default;
#endif

void RemoteInspector::startDisabled()
{
    RemoteInspector::startEnabled = false;
}

TargetID RemoteInspector::nextAvailableTargetIdentifier()
{
    TargetID nextValidTargetIdentifier;
    do {
        nextValidTargetIdentifier = m_nextAvailableTargetIdentifier++;
    } while (!nextValidTargetIdentifier || nextValidTargetIdentifier == std::numeric_limits<TargetID>::max() || m_targetMap.contains(nextValidTargetIdentifier));
    return nextValidTargetIdentifier;
}

void RemoteInspector::registerTarget(RemoteControllableTarget* target)
{
    ASSERT_ARG(target, target);

    Locker locker { m_mutex };

    auto targetIdentifier = nextAvailableTargetIdentifier();
    target->setTargetIdentifier(targetIdentifier);

    {
        auto result = m_targetMap.set(targetIdentifier, target);
        ASSERT_UNUSED(result, result.isNewEntry);
    }

    // If remote control is not allowed, a null listing is returned.
    if (auto targetListing = listingForTarget(*target)) {
        auto result = m_targetListingMap.set(targetIdentifier, targetListing);
        ASSERT_UNUSED(result, result.isNewEntry);
    }

    pushListingsSoon();
}

void RemoteInspector::unregisterTarget(RemoteControllableTarget* target)
{
    ASSERT_ARG(target, target);

    Locker locker { m_mutex };

    auto targetIdentifier = target->targetIdentifier();
    if (!targetIdentifier)
        return;

    bool wasRemoved = m_targetMap.remove(targetIdentifier);
    ASSERT_UNUSED(wasRemoved, wasRemoved);

    // The listing may never have been added if remote control isn't allowed.
    m_targetListingMap.remove(targetIdentifier);

    if (auto connectionToTarget = m_targetConnectionMap.take(targetIdentifier))
        connectionToTarget->targetClosed();

    pushListingsSoon();
}

void RemoteInspector::updateTarget(RemoteControllableTarget* target)
{
    ASSERT_ARG(target, target);

    Locker locker { m_mutex };

    if (!updateTargetMap(target))
        return;

    pushListingsSoon();
}

bool RemoteInspector::updateTargetMap(RemoteControllableTarget* target)
{
    ASSERT(m_mutex.isLocked());

    auto targetIdentifier = target->targetIdentifier();
    if (!targetIdentifier)
        return false;

    auto result = m_targetMap.set(targetIdentifier, target);
    ASSERT_UNUSED(result, !result.isNewEntry);

    // If the target has just allowed remote control, then the listing won't exist yet.
    // If the target has no identifier remove the old listing.
    if (auto targetListing = listingForTarget(*target))
        m_targetListingMap.set(targetIdentifier, targetListing);
    else
        m_targetListingMap.remove(targetIdentifier);

    return true;
}

#if !PLATFORM(COCOA)
void RemoteInspector::updateAutomaticInspectionCandidate(RemoteInspectionTarget* target)
{
    // FIXME: Implement automatic inspection.
    updateTarget(target);
}
#endif

void RemoteInspector::updateClientCapabilities()
{
    ASSERT(isMainThread());

    Locker locker { m_mutex };

    if (!m_client)
        m_clientCapabilities = std::nullopt;
    else {
        RemoteInspector::Client::Capabilities updatedCapabilities = {
            m_client->remoteAutomationAllowed(),
            m_client->browserName(),
            m_client->browserVersion()
        };

        m_clientCapabilities = updatedCapabilities;
    }
}

void RemoteInspector::setClient(RemoteInspector::Client* client)
{
    ASSERT((m_client && !client) || (!m_client && client));

    {
        Locker locker { m_mutex };
        m_client = client;
    }

    // Send an updated listing that includes whether the client allows remote automation.
    updateClientCapabilities();
    pushListingsSoon();
}

void RemoteInspector::setupFailed(TargetID targetIdentifier)
{
    Locker locker { m_mutex };

    m_targetConnectionMap.remove(targetIdentifier);
    m_pausedAutomaticInspectionCandidates.remove(targetIdentifier);

    updateHasActiveDebugSession();
    updateTargetListing(targetIdentifier);
    pushListingsSoon();
}

void RemoteInspector::setupCompleted(TargetID targetIdentifier)
{
    Locker locker { m_mutex };

    m_pausedAutomaticInspectionCandidates.remove(targetIdentifier);
}

bool RemoteInspector::waitingForAutomaticInspection(TargetID targetIdentifier)
{
    Locker locker { m_mutex };

    return m_pausedAutomaticInspectionCandidates.contains(targetIdentifier);
}

void RemoteInspector::clientCapabilitiesDidChange()
{
    updateClientCapabilities();
    pushListingsSoon();
}

void RemoteInspector::stop()
{
    Locker locker { m_mutex };

    stopInternal(StopSource::API);
}

TargetListing RemoteInspector::listingForTarget(const RemoteControllableTarget& target) const
{
    if (auto* inspectionTarget = dynamicDowncast<RemoteInspectionTarget>(target))
        return listingForInspectionTarget(*inspectionTarget);
    if (auto* automationTarget = dynamicDowncast<RemoteAutomationTarget>(target))
        return listingForAutomationTarget(*automationTarget);

    ASSERT_NOT_REACHED();
    return nullptr;
}

void RemoteInspector::updateTargetListing(TargetID targetIdentifier)
{
    auto target = m_targetMap.get(targetIdentifier);
    if (!target)
        return;

    updateTargetListing(*target);
}

void RemoteInspector::updateTargetListing(const RemoteControllableTarget& target)
{
    auto targetListing = listingForTarget(target);
    if (!targetListing)
        return;

    m_targetListingMap.set(target.targetIdentifier(), targetListing);

    pushListingsSoon();
}

void RemoteInspector::updateHasActiveDebugSession()
{
    bool hasActiveDebuggerSession = !m_targetConnectionMap.isEmpty();
    if (hasActiveDebuggerSession == m_hasActiveDebugSession)
        return;

    m_hasActiveDebugSession = hasActiveDebuggerSession;

    // FIXME: Expose some way to access this state in an embedder.
    // Legacy iOS WebKit 1 had a notification. This will need to be smarter with WebKit2.
}

RemoteInspector::Client::Client() = default;
RemoteInspector::Client::~Client() = default;

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
