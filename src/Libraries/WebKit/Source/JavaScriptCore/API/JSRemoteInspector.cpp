/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "JSRemoteInspector.h"

#include "JSGlobalObjectConsoleClient.h"
#include <wtf/ProcessID.h>

#if ENABLE(REMOTE_INSPECTOR)
#include "RemoteInspector.h"
#endif

#if PLATFORM(COCOA)
#include <wtf/cocoa/Entitlements.h>
#include <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

using namespace Inspector;

static std::optional<bool> remoteInspectionEnabledByDefault = std::nullopt;
static bool inspectionFollowsInternalPolicies = true;

void JSRemoteInspectorDisableAutoStart(void)
{
#if ENABLE(REMOTE_INSPECTOR)
    RemoteInspector::startDisabled();
#endif
}

void JSRemoteInspectorStart(void)
{
#if ENABLE(REMOTE_INSPECTOR)
    RemoteInspector::singleton();
#endif
}

void JSRemoteInspectorSetParentProcessInformation(ProcessID pid, const uint8_t* auditData, size_t auditLength)
{
#if ENABLE(REMOTE_INSPECTOR) && PLATFORM(COCOA)
    RetainPtr<CFDataRef> auditDataRef = adoptCF(CFDataCreate(kCFAllocatorDefault, auditData, auditLength));
    RemoteInspector::singleton().setParentProcessInformation(pid, auditDataRef);
#else
    UNUSED_PARAM(pid);
    UNUSED_PARAM(auditData);
    UNUSED_PARAM(auditLength);
#endif
}

void JSRemoteInspectorSetLogToSystemConsole(bool logToSystemConsole)
{
    JSGlobalObjectConsoleClient::setLogToSystemConsole(logToSystemConsole);
}

#if PLATFORM(COCOA)
static bool mainProcessHasEntitlement(ASCIILiteral entitlement, std::optional<audit_token_t> parentProcessAuditToken)
{
    if (parentProcessAuditToken)
        return WTF::hasEntitlement(*parentProcessAuditToken, entitlement);

    return WTF::processHasEntitlement(entitlement);
}
#endif

static bool defaultStateForRemoteInspectionEnabledByDefault(void)
{
#if PLATFORM(COCOA)
    auto parentProcessAuditToken = RemoteInspector::singleton().parentProcessAuditToken();

    if (!linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::InspectableDefaultsToDisabled)) {
#if PLATFORM(MAC)
        auto developerProvisioningEntitlement = "com.apple.security.get-task-allow"_s;
#else
        auto developerProvisioningEntitlement = "get-task-allow"_s;
#endif
        if (mainProcessHasEntitlement(developerProvisioningEntitlement, parentProcessAuditToken)) {
            WTFLogAlways("Inspection is enabled by default for process or parent application with '%s' entitlement linked against old SDK. Use `inspectable` API to enable inspection on newer SDKs.", developerProvisioningEntitlement.characters());
            return true;
        }
    }

#if PLATFORM(MAC)
    auto deprecatedWebInspectorAllowEntitlement = "com.apple.webinspector.allow"_s;
#else
    auto deprecatedWebInspectorAllowEntitlement = "com.apple.private.webinspector.allow-remote-inspection"_s;
#endif
    if (mainProcessHasEntitlement(deprecatedWebInspectorAllowEntitlement, parentProcessAuditToken)) {
        WTFLogAlways("Inspection is enabled by default for process or parent application with deprecated '%s' entitlement. Use `inspectable` API to enable inspection instead.", deprecatedWebInspectorAllowEntitlement.characters());
        return true;
    }

    return false;
#else
    return true;
#endif // not PLATFORM(COCOA)
}

bool JSRemoteInspectorGetInspectionEnabledByDefault(void)
{
    if (!remoteInspectionEnabledByDefault)
        remoteInspectionEnabledByDefault = defaultStateForRemoteInspectionEnabledByDefault();

    return remoteInspectionEnabledByDefault.value();
}

void JSRemoteInspectorSetInspectionEnabledByDefault(bool enabledByDefault)
{
    remoteInspectionEnabledByDefault = enabledByDefault;
}

bool JSRemoteInspectorGetInspectionFollowsInternalPolicies(void)
{
    return inspectionFollowsInternalPolicies;
}

void JSRemoteInspectorSetInspectionFollowsInternalPolicies(bool followsInternalPolicies)
{
    inspectionFollowsInternalPolicies = followsInternalPolicies;
}
