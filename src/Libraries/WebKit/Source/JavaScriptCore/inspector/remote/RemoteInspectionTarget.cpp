/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#include "RemoteInspectionTarget.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "JSGlobalObjectDebugger.h"
#include "RemoteInspector.h"
#include <wtf/RunLoop.h>

#if PLATFORM(COCOA)
#include <wtf/spi/darwin/OSVariantSPI.h>
#endif

namespace Inspector {

RemoteInspectionTarget::RemoteInspectionTarget() = default;

RemoteInspectionTarget::~RemoteInspectionTarget() = default;

bool RemoteInspectionTarget::remoteControlAllowed() const
{
    return allowsInspectionByPolicy() || hasLocalDebugger();
}

bool RemoteInspectionTarget::allowsInspectionByPolicy() const
{
    switch (m_inspectable) {
    case Inspectable::Yes:
        return true;
    case Inspectable::No:
#if PLATFORM(COCOA)
        static bool allowInternalSecurityPolicies = os_variant_allows_internal_security_policies("com.apple.WebInspector");
        if (allowInternalSecurityPolicies && !RemoteInspector::singleton().isSimulatingCustomerInstall())
            return true;
        FALLTHROUGH;
#endif
    case Inspectable::NoIgnoringInternalPolicies:
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

bool RemoteInspectionTarget::inspectable() const
{
    switch (m_inspectable) {
    case Inspectable::Yes:
        return true;
    case Inspectable::No:
    case Inspectable::NoIgnoringInternalPolicies:
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

void RemoteInspectionTarget::setInspectable(bool inspectable)
{
    if (inspectable)
        m_inspectable = Inspectable::Yes;
    else {
        if (!JSRemoteInspectorGetInspectionFollowsInternalPolicies())
            m_inspectable = Inspectable::NoIgnoringInternalPolicies;
        else
            m_inspectable = Inspectable::No;
    }

    if (allowsInspectionByPolicy() && automaticInspectionAllowed())
        RemoteInspector::singleton().updateAutomaticInspectionCandidate(this);
    else
        RemoteInspector::singleton().updateTarget(this);
}

void RemoteInspectionTarget::pauseWaitingForAutomaticInspection()
{
    ASSERT(targetIdentifier());
    ASSERT(allowsInspectionByPolicy());
    ASSERT(automaticInspectionAllowed());

    while (RemoteInspector::singleton().waitingForAutomaticInspection(targetIdentifier())) {
        if (RunLoop::cycle(JSGlobalObjectDebugger::runLoopMode()) == RunLoop::CycleResult::Stop)
            break;
    }
}

void RemoteInspectionTarget::unpauseForInitializedInspector()
{
    RemoteInspector::singleton().setupCompleted(targetIdentifier());
}

void RemoteInspectionTarget::setPresentingApplicationPID(std::optional<ProcessID>&& pid)
{
    m_presentingApplicationPID = pid;
#if PLATFORM(COCOA)
    RemoteInspector::singleton().setUsePerTargetPresentingApplicationPIDs(true);
#endif
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
