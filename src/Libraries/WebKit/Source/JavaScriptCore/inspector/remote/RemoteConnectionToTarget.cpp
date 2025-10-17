/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#include "RemoteConnectionToTarget.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteAutomationTarget.h"
#include "RemoteInspectionTarget.h"
#include "RemoteInspector.h"
#include <wtf/RunLoop.h>

namespace Inspector {

RemoteConnectionToTarget::RemoteConnectionToTarget(RemoteControllableTarget& target)
    : m_target(&target)
{
}

RemoteConnectionToTarget::~RemoteConnectionToTarget() = default;

bool RemoteConnectionToTarget::setup(bool isAutomaticInspection, bool automaticallyPause)
{
    RefPtr<RemoteControllableTarget> target;
    TargetID targetIdentifier;

    {
        Locker locker { m_targetMutex };
        target = m_target.get();
        if (!target)
            return false;
        targetIdentifier = this->targetIdentifier().value_or(0);
    }

    if (!target->remoteControlAllowed()) {
        RemoteInspector::singleton().setupFailed(targetIdentifier);
        Locker locker { m_targetMutex };
        m_target = nullptr;
    } else if (auto* inspectionTarget = dynamicDowncast<RemoteInspectionTarget>(*target)) {
        inspectionTarget->connect(*this, isAutomaticInspection, automaticallyPause);
        m_connected = true;

        RemoteInspector::singleton().updateTargetListing(targetIdentifier);
    } else if (auto* inspectionTarget = dynamicDowncast<RemoteAutomationTarget>(*target)) {
        inspectionTarget->connect(*this);
        m_connected = true;

        RemoteInspector::singleton().updateTargetListing(targetIdentifier);
    }

    return true;
}

void RemoteConnectionToTarget::sendMessageToTarget(String&& message)
{
    RefPtr<RemoteControllableTarget> target;
    {
        Locker locker { m_targetMutex };
        target = m_target.get();
    }
    if (target)
        target->dispatchMessageFromRemote(WTFMove(message));
}

void RemoteConnectionToTarget::close()
{
    RunLoop::current().dispatch([this, protectThis = Ref { *this }] {
        Locker locker { m_targetMutex };
        TargetID targetIdentifier = 0;

        if (RefPtr target = m_target.get()) {
            targetIdentifier = target->targetIdentifier();

            if (m_connected)
                target->disconnect(*this);

            m_target = nullptr;
        }

        if (targetIdentifier)
            RemoteInspector::singleton().updateTargetListing(targetIdentifier);
    });
}

void RemoteConnectionToTarget::targetClosed()
{
    Locker locker { m_targetMutex };
    m_target = nullptr;
}

std::optional<TargetID> RemoteConnectionToTarget::targetIdentifier() const
{
    RefPtr target = m_target.get();
    return target ? std::optional<TargetID>(target->targetIdentifier()) : std::nullopt;
}

void RemoteConnectionToTarget::sendMessageToFrontend(const String& message)
{
    std::optional<TargetID> targetIdentifier;
    {
        Locker locker { m_targetMutex };
        RefPtr target = m_target.get();
        if (!target)
            return;
        targetIdentifier = target->targetIdentifier();
    }
    RemoteInspector::singleton().sendMessageToRemote(*targetIdentifier, message);
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
