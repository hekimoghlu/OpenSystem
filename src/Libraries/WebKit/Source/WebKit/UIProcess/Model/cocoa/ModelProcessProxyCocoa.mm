/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "ModelProcessProxy.h"

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS) && ENABLE(MODEL_PROCESS)

#include "GPUProcessProxy.h"
#include "RunningBoardServicesSPI.h"
#include "SharedFileHandle.h"
#include "SharedPreferencesForWebProcess.h"
#include "WebProcessProxy.h"

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, connection())

namespace WebKit {

void ModelProcessProxy::requestSharedSimulationConnection(WebCore::ProcessIdentifier webProcessIdentifier, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&& completionHandler)
{
    auto webProcessProxy = WebProcessProxy::processForIdentifier(webProcessIdentifier);
    if (!webProcessProxy) {
        RELEASE_LOG_ERROR(Process, "%p - ModelProcessProxy::requestSharedSimulationConnection() No WebProcessProxy with this identifier", this);
        completionHandler(std::nullopt);
        return;
    }

    MESSAGE_CHECK(webProcessProxy->sharedPreferencesForWebProcessValue().modelElementEnabled);
    MESSAGE_CHECK(webProcessProxy->sharedPreferencesForWebProcessValue().modelProcessEnabled);
    // We only expect shared simulation connection to be set up once for each model process instance.
    MESSAGE_CHECK(!m_didInitializeSharedSimulationConnection);

    m_didInitializeSharedSimulationConnection = true;

    NSError *error;
    RBSProcessHandle *process = [RBSProcessHandle handleForIdentifier:[RBSProcessIdentifier identifierWithPid:processID()] error:&error];
    if (error) {
        RELEASE_LOG_ERROR(ModelElement, "%p - ModelProcessProxy: Failed to get audit token for requesting process: %@", this, error);
        completionHandler(std::nullopt);
        return;
    }

    RELEASE_LOG(ModelElement, "%p - ModelProcessProxy: Requesting shared simulation connection for model process with audit token for pid=%d", this, processID());
    GPUProcessProxy::getOrCreate()->requestSharedSimulationConnection(process.auditToken, WTFMove(completionHandler));
}

}

#undef MESSAGE_CHECK

#endif

