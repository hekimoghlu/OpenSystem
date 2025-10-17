/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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

#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "InspectorFrontendDispatchers.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace Inspector {

class BackendDispatcher;
class InspectorEnvironment;

class JS_EXPORT_PRIVATE InspectorAgent final : public InspectorAgentBase, public InspectorBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorAgent);
public:
    InspectorAgent(AgentContext&);
    ~InspectorAgent() final;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // InspectorBackendDispatcherHandler
    Protocol::ErrorStringOr<void> enable() final;
    Protocol::ErrorStringOr<void> disable() final;
    Protocol::ErrorStringOr<void> initialized() final;

    // CommandLineAPI
    void inspect(Ref<Protocol::Runtime::RemoteObject>&&, Ref<JSON::Object>&& hints);

    void evaluateForTestInFrontend(const String& script);

private:
    InspectorEnvironment& m_environment;
    std::unique_ptr<InspectorFrontendDispatcher> m_frontendDispatcher;
    Ref<InspectorBackendDispatcher> m_backendDispatcher;

    Vector<String> m_pendingEvaluateTestCommands;
    std::pair<RefPtr<Protocol::Runtime::RemoteObject>, RefPtr<JSON::Object>> m_pendingInspectData;
    bool m_enabled { false };
};

} // namespace Inspector
