/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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

#include <optional>
#include <wtf/Forward.h>
#include <wtf/ProcessID.h>

namespace WTF {

WTF_EXPORT_PRIVATE void setLegacyPresentingApplicationPID(int);
WTF_EXPORT_PRIVATE int legacyPresentingApplicationPID();

#if HAVE(AUDIT_TOKEN)
WTF_EXPORT_PRIVATE ProcessID pidFromAuditToken(const audit_token_t&);
#endif

enum class AuxiliaryProcessType : uint8_t {
    WebContent,
    Network,
    Plugin,
#if ENABLE(GPU_PROCESS)
    GPU,
#endif
#if ENABLE(MODEL_PROCESS)
    Model,
#endif
};

WTF_EXPORT_PRIVATE void setAuxiliaryProcessType(AuxiliaryProcessType);
WTF_EXPORT_PRIVATE void setAuxiliaryProcessTypeForTesting(std::optional<AuxiliaryProcessType>);
WTF_EXPORT_PRIVATE bool checkAuxiliaryProcessType(AuxiliaryProcessType);
WTF_EXPORT_PRIVATE std::optional<AuxiliaryProcessType> processType();
WTF_EXPORT_PRIVATE ASCIILiteral processTypeDescription(std::optional<AuxiliaryProcessType>);

WTF_EXPORT_PRIVATE bool isInAuxiliaryProcess();
inline bool isInWebProcess() { return checkAuxiliaryProcessType(AuxiliaryProcessType::WebContent); }
inline bool isInNetworkProcess() { return checkAuxiliaryProcessType(AuxiliaryProcessType::Network); }
inline bool isInGPUProcess()
{
#if ENABLE(GPU_PROCESS)
    return checkAuxiliaryProcessType(AuxiliaryProcessType::GPU);
#else
    return false;
#endif
}

inline bool isInModelProcess()
{
#if ENABLE(MODEL_PROCESS)
    return checkAuxiliaryProcessType(AuxiliaryProcessType::Model);
#else
    return false;
#endif
}

} // namespace WTF

using WTF::checkAuxiliaryProcessType;
using WTF::isInAuxiliaryProcess;
using WTF::isInGPUProcess;
using WTF::isInModelProcess;
using WTF::isInNetworkProcess;
using WTF::isInWebProcess;
using WTF::legacyPresentingApplicationPID;
using WTF::processType;
using WTF::processTypeDescription;
using WTF::setAuxiliaryProcessType;
using WTF::setAuxiliaryProcessTypeForTesting;
using WTF::setLegacyPresentingApplicationPID;

#if HAVE(AUDIT_TOKEN)
using WTF::pidFromAuditToken;
#endif
