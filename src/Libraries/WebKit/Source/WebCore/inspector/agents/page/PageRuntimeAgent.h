/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <JavaScriptCore/InspectorRuntimeAgent.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class CallFrame;
}

namespace WebCore {

class DOMWrapperWorld;
class LocalFrame;
class Page;
class SecurityOrigin;

class PageRuntimeAgent final : public Inspector::InspectorRuntimeAgent {
    WTF_MAKE_NONCOPYABLE(PageRuntimeAgent);
    WTF_MAKE_TZONE_ALLOCATED(PageRuntimeAgent);
public:
    PageRuntimeAgent(PageAgentContext&);
    ~PageRuntimeAgent();

    // RuntimeBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<std::tuple<Ref<Inspector::Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> evaluate(const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture);
    void callFunctionOn(const Inspector::Protocol::Runtime::RemoteObjectId&, const String& functionDeclaration, RefPtr<JSON::Array>&& arguments, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& emulateUserGesture, std::optional<bool>&& awaitPromise, Ref<CallFunctionOnCallback>&&);

    // InspectorInstrumentation
    void frameNavigated(LocalFrame&);
    void didClearWindowObjectInWorld(LocalFrame&, DOMWrapperWorld&);

private:
    Inspector::InjectedScript injectedScriptForEval(Inspector::Protocol::ErrorString&, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&);
    void muteConsole();
    void unmuteConsole();
    void reportExecutionContextCreation();
    void notifyContextCreated(const Inspector::Protocol::Network::FrameId&, JSC::JSGlobalObject*, const DOMWrapperWorld&, SecurityOrigin* = nullptr);

    std::unique_ptr<Inspector::RuntimeFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::RuntimeBackendDispatcher> m_backendDispatcher;

    InstrumentingAgents& m_instrumentingAgents;

    WeakRef<Page> m_inspectedPage;
};

} // namespace WebCore
