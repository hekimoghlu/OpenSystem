/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
#include "InspectorAgentRegistry.h"

#include "InspectorAgentBase.h"

namespace Inspector {

AgentRegistry::AgentRegistry() = default;

AgentRegistry::~AgentRegistry()
{
    // Allow agents to remove cross-references to other agents that would otherwise
    // make it difficult to establish a correct destruction order for all agents.
    for (auto& agent : m_agents)
        agent->discardAgent();
}

void AgentRegistry::append(std::unique_ptr<InspectorAgentBase> agent)
{
    m_agents.append(WTFMove(agent));
}

void AgentRegistry::didCreateFrontendAndBackend(FrontendRouter* frontendRouter, BackendDispatcher* backendDispatcher)
{
    for (auto& agent : m_agents)
        agent->didCreateFrontendAndBackend(frontendRouter, backendDispatcher);
}

void AgentRegistry::willDestroyFrontendAndBackend(DisconnectReason reason)
{
    for (auto& agent : m_agents)
        agent->willDestroyFrontendAndBackend(reason);
}

void AgentRegistry::discardValues()
{
    for (auto& agent : m_agents)
        agent->discardValues();
}

} // namespace Inspector
