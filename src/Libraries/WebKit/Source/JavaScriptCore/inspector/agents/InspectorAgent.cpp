/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include "InspectorAgent.h"

#include "InspectorEnvironment.h"
#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorAgent);

InspectorAgent::InspectorAgent(AgentContext& context)
    : InspectorAgentBase("Inspector"_s)
    , m_environment(context.environment)
    , m_frontendDispatcher(makeUnique<InspectorFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(InspectorBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorAgent::~InspectorAgent() = default;

void InspectorAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    m_pendingEvaluateTestCommands.clear();

    disable();
}

Protocol::ErrorStringOr<void> InspectorAgent::enable()
{
    m_enabled = true;

    if (m_pendingInspectData.first)
        inspect(m_pendingInspectData.first.releaseNonNull(), m_pendingInspectData.second.releaseNonNull());

    for (auto& testCommand : m_pendingEvaluateTestCommands)
        m_frontendDispatcher->evaluateForTestInFrontend(testCommand);

    m_pendingEvaluateTestCommands.clear();

    return { };
}

Protocol::ErrorStringOr<void> InspectorAgent::disable()
{
    m_enabled = false;

    return { };
}

Protocol::ErrorStringOr<void> InspectorAgent::initialized()
{
    m_environment.frontendInitialized();

    return { };
}

void InspectorAgent::inspect(Ref<Protocol::Runtime::RemoteObject>&& object, Ref<JSON::Object>&& hints)
{
    if (m_enabled) {
        m_frontendDispatcher->inspect(WTFMove(object), WTFMove(hints));
        m_pendingInspectData.first = nullptr;
        m_pendingInspectData.second = nullptr;
        return;
    }

    m_pendingInspectData.first = WTFMove(object);
    m_pendingInspectData.second = WTFMove(hints);
}

void InspectorAgent::evaluateForTestInFrontend(const String& script)
{
    if (m_enabled)
        m_frontendDispatcher->evaluateForTestInFrontend(script);
    else
        m_pendingEvaluateTestCommands.append(script);
}

} // namespace Inspector
