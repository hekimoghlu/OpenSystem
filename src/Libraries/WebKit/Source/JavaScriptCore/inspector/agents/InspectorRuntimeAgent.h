/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class Debugger;
class VM;
}

namespace Inspector {

class InjectedScript;
class InjectedScriptManager;

class JS_EXPORT_PRIVATE InspectorRuntimeAgent : public InspectorAgentBase, public RuntimeBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorRuntimeAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorRuntimeAgent);
public:
    ~InspectorRuntimeAgent() override;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // RuntimeBackendDispatcherHandler
    Protocol::ErrorStringOr<void> enable() override;
    Protocol::ErrorStringOr<void> disable() override;
    Protocol::ErrorStringOr<std::tuple<Protocol::Runtime::SyntaxErrorType, String /* message */, RefPtr<Protocol::Runtime::ErrorRange>>> parse(const String& expression) final;
    Protocol::ErrorStringOr<std::tuple<Ref<Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> evaluate(const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<Protocol::Runtime::ExecutionContextId>&&, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture) override;
    void awaitPromise(const Protocol::Runtime::RemoteObjectId&, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, Ref<AwaitPromiseCallback>&&) final;
    void callFunctionOn(const Protocol::Runtime::RemoteObjectId&, const String& functionDeclaration, RefPtr<JSON::Array>&& arguments, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& emulateUserGesture, std::optional<bool>&& awaitPromise, Ref<CallFunctionOnCallback>&&) override;
    Protocol::ErrorStringOr<void> releaseObject(const Protocol::Runtime::RemoteObjectId&) final;
    Protocol::ErrorStringOr<Ref<Protocol::Runtime::ObjectPreview>> getPreview(const Protocol::Runtime::RemoteObjectId&) final;
    Protocol::ErrorStringOr<std::tuple<Ref<JSON::ArrayOf<Protocol::Runtime::PropertyDescriptor>>, RefPtr<JSON::ArrayOf<Protocol::Runtime::InternalPropertyDescriptor>>>> getProperties(const Protocol::Runtime::RemoteObjectId&, std::optional<bool>&& ownProperties, std::optional<int>&& fetchStart, std::optional<int>&& fetchCount, std::optional<bool>&& generatePreview) final;
    Protocol::ErrorStringOr<std::tuple<Ref<JSON::ArrayOf<Protocol::Runtime::PropertyDescriptor>>, RefPtr<JSON::ArrayOf<Protocol::Runtime::InternalPropertyDescriptor>>>> getDisplayableProperties(const Protocol::Runtime::RemoteObjectId&, std::optional<int>&& fetchStart, std::optional<int>&& fetchCount, std::optional<bool>&& generatePreview) final;
    Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Protocol::Runtime::CollectionEntry>>> getCollectionEntries(const Protocol::Runtime::RemoteObjectId&, const String& objectGroup, std::optional<int>&& fetchStart, std::optional<int>&& fetchCount) final;
    Protocol::ErrorStringOr<std::optional<int> /* saveResultIndex */> saveResult(Ref<JSON::Object>&& callArgument, std::optional<Protocol::Runtime::ExecutionContextId>&&) final;
    Protocol::ErrorStringOr<void> setSavedResultAlias(const String&) final;
    Protocol::ErrorStringOr<void> releaseObjectGroup(const String&) final;
    Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Protocol::Runtime::TypeDescription>>> getRuntimeTypesForVariablesAtOffsets(Ref<JSON::Array>&& locations) final;
    Protocol::ErrorStringOr<void> enableTypeProfiler() final;
    Protocol::ErrorStringOr<void> disableTypeProfiler() final;
    Protocol::ErrorStringOr<void> enableControlFlowProfiler() final;
    Protocol::ErrorStringOr<void> disableControlFlowProfiler() final;
    Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Protocol::Runtime::BasicBlock>>> getBasicBlocks(const String& sourceID) final;

protected:
    InspectorRuntimeAgent(AgentContext&);

    Protocol::ErrorStringOr<std::tuple<Ref<Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> evaluate(InjectedScript&, const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture);
    void callFunctionOn(InjectedScript&, const Protocol::Runtime::RemoteObjectId&, const String& functionDeclaration, RefPtr<JSON::Array>&& arguments, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& emulateUserGesture, std::optional<bool>&& awaitPromise, Ref<CallFunctionOnCallback>&&);

    InjectedScriptManager& injectedScriptManager() { return m_injectedScriptManager; }

    virtual InjectedScript injectedScriptForEval(Protocol::ErrorString&, std::optional<Protocol::Runtime::ExecutionContextId>&&) = 0;

    virtual void muteConsole() = 0;
    virtual void unmuteConsole() = 0;

private:
    void setTypeProfilerEnabledState(bool);
    void setControlFlowProfilerEnabledState(bool);

    InjectedScriptManager& m_injectedScriptManager;
    JSC::Debugger& m_debugger;
    JSC::VM& m_vm;
    bool m_enabled {false};
    bool m_isTypeProfilingEnabled {false};
    bool m_isControlFlowProfilingEnabled {false};
};

} // namespace Inspector
