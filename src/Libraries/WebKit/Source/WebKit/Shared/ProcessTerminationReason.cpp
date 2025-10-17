/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include "ProcessTerminationReason.h"

#include <wtf/text/ASCIILiteral.h>

namespace WebKit {

ASCIILiteral processTerminationReasonToString(ProcessTerminationReason reason)
{
    switch (reason) {
    case ProcessTerminationReason::ExceededMemoryLimit:
        return "ExceededMemoryLimit"_s;
    case ProcessTerminationReason::ExceededCPULimit:
        return "ExceededCPULimit"_s;
    case ProcessTerminationReason::RequestedByClient:
        return "RequestedByClient"_s;
    case ProcessTerminationReason::IdleExit:
        return "IdleExit"_s;
    case ProcessTerminationReason::Unresponsive:
        return "Unresponsive"_s;
    case ProcessTerminationReason::Crash:
        return "Crash"_s;
    case ProcessTerminationReason::ExceededProcessCountLimit:
        return "ExceededProcessCountLimit"_s;
    case ProcessTerminationReason::NavigationSwap:
        return "NavigationSwap"_s;
    case ProcessTerminationReason::RequestedByNetworkProcess:
        return "RequestedByNetworkProcess"_s;
    case ProcessTerminationReason::RequestedByGPUProcess:
        return "RequestedByGPUProcess"_s;
    case ProcessTerminationReason::RequestedByModelProcess:
        return "RequestedByModelProcess"_s;
    case ProcessTerminationReason::GPUProcessCrashedTooManyTimes:
        return "GPUProcessCrashedTooManyTimes"_s;
    case ProcessTerminationReason::ModelProcessCrashedTooManyTimes:
        return "ModelProcessCrashedTooManyTimes"_s;
    case ProcessTerminationReason::NonMainFrameWebContentProcessCrash:
        return "NonMainFrameWebContentProcessCrash"_s;
    }

    return ""_s;
}

}
