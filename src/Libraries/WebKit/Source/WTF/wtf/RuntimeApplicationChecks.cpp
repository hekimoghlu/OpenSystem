/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#include "RuntimeApplicationChecks.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/ProcessID.h>
#include <wtf/RunLoop.h>

namespace WTF {

#if !ASSERT_MSG_DISABLED
static bool presentingApplicationPIDOverrideWasQueried;
#endif

static std::optional<int>& presentingApplicationPIDOverride()
{
    static NeverDestroyed<std::optional<int>> pid;
#if !ASSERT_MSG_DISABLED
    presentingApplicationPIDOverrideWasQueried = true;
#endif
    return pid;
}

int legacyPresentingApplicationPID()
{
    const auto& pid = presentingApplicationPIDOverride();
    ASSERT(!pid || RunLoop::isMain());
    return pid ? pid.value() : getCurrentProcessID();
}

void setLegacyPresentingApplicationPID(int pid)
{
    ASSERT(RunLoop::isMain());
    ASSERT_WITH_MESSAGE(!presentingApplicationPIDOverrideWasQueried, "legacyPresentingApplicationPID() should not be called before setLegacyPresentingApplicationPID()");
    presentingApplicationPIDOverride() = pid;
}

#if HAVE(AUDIT_TOKEN)
ProcessID pidFromAuditToken(const audit_token_t& auditToken)
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    return auditToken.val[5];
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}
#endif

static std::optional<AuxiliaryProcessType>& auxiliaryProcessType()
{
    static std::optional<AuxiliaryProcessType> auxiliaryProcessType;
    return auxiliaryProcessType;
}

bool isInAuxiliaryProcess()
{
    return !!auxiliaryProcessType();
}

void setAuxiliaryProcessType(AuxiliaryProcessType type)
{
    auxiliaryProcessType() = type;
}

void setAuxiliaryProcessTypeForTesting(std::optional<AuxiliaryProcessType> type)
{
    auxiliaryProcessType() = type;
}

bool checkAuxiliaryProcessType(AuxiliaryProcessType type)
{
    auto currentType = auxiliaryProcessType();
    if (!currentType)
        return false;
    return *currentType == type;
}

std::optional<AuxiliaryProcessType> processType()
{
    return auxiliaryProcessType();
}

ASCIILiteral processTypeDescription(std::optional<AuxiliaryProcessType> type)
{
    if (!type)
        return "UI"_s;

    switch (*type) {
    case AuxiliaryProcessType::WebContent:
        return "Web"_s;
    case AuxiliaryProcessType::Network:
        return "Network"_s;
    case AuxiliaryProcessType::Plugin:
        return "Plugin"_s;
#if ENABLE(GPU_PROCESS)
    case AuxiliaryProcessType::GPU:
        return "GPU"_s;
#endif
#if ENABLE(MODEL_PROCESS)
    case AuxiliaryProcessType::Model:
        return "Model"_s;
#endif
    }
    return "Unknown"_s;
}

} // namespace WTF
