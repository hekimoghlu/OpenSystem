/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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

#include "InspectorWebAgentBase.h"
#include <JavaScriptCore/InspectorRuntimeAgent.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WorkerOrWorkletGlobalScope;

class WorkerRuntimeAgent final : public Inspector::InspectorRuntimeAgent {
    WTF_MAKE_NONCOPYABLE(WorkerRuntimeAgent);
    WTF_MAKE_TZONE_ALLOCATED(WorkerRuntimeAgent);
public:
    WorkerRuntimeAgent(WorkerAgentContext&);
    ~WorkerRuntimeAgent();

private:
    Inspector::InjectedScript injectedScriptForEval(Inspector::Protocol::ErrorString&, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&);

    // We don't need to mute console for workers.
    void muteConsole() { }
    void unmuteConsole() { }

    RefPtr<Inspector::RuntimeBackendDispatcher> m_backendDispatcher;
    WorkerOrWorkletGlobalScope& m_globalScope;
};

} // namespace WebCore
