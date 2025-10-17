/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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

#include "WebDebuggerAgent.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DOMWrapperWorld;
class Document;
class LocalFrame;
class Page;
class UserGestureEmulationScope;

class PageDebuggerAgent final : public WebDebuggerAgent {
    WTF_MAKE_NONCOPYABLE(PageDebuggerAgent);
    WTF_MAKE_TZONE_ALLOCATED(PageDebuggerAgent);
public:
    PageDebuggerAgent(PageAgentContext&);
    ~PageDebuggerAgent();
    bool enabled() const;

    // DebuggerBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<std::tuple<Ref<Inspector::Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> evaluateOnCallFrame(const Inspector::Protocol::Debugger::CallFrameId&, const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture);

    // JSC::Debugger::Client
    void debuggerWillEvaluate(JSC::Debugger&, JSC::JSGlobalObject*, const JSC::Breakpoint::Action&);
    void debuggerDidEvaluate(JSC::Debugger&, JSC::JSGlobalObject*, const JSC::Breakpoint::Action&);

    // JSC::Debugger::Observer
    void breakpointActionLog(JSC::JSGlobalObject*, const String& data);

    // InspectorInstrumentation
    void didClearWindowObjectInWorld(LocalFrame&, DOMWrapperWorld&);
    void mainFrameStartedLoading();
    void mainFrameStoppedLoading();
    void mainFrameNavigated();
    void didRequestAnimationFrame(int callbackId, Document&);
    void willFireAnimationFrame(int callbackId);
    void didCancelAnimationFrame(int callbackId);
    void didFireAnimationFrame(int callbackId);

private:
    void internalEnable();
    void internalDisable(bool isBeingDestroyed);

    String sourceMapURLForScript(const JSC::Debugger::Script&);

    void muteConsole();
    void unmuteConsole();

    Inspector::InjectedScript injectedScriptForEval(Inspector::Protocol::ErrorString&, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&);

    WeakRef<Page> m_inspectedPage;
    Vector<UniqueRef<UserGestureEmulationScope>> m_breakpointActionUserGestureEmulationScopeStack;
};

} // namespace WebCore
