/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include "PageDebuggerAgent.h"

#include "CachedResource.h"
#include "DOMWrapperWorld.h"
#include "Document.h"
#include "InspectorPageAgent.h"
#include "InstrumentingAgents.h"
#include "JSDOMWindowCustom.h"
#include "JSExecState.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include "PageDebugger.h"
#include "ScriptExecutionContext.h"
#include "UserGestureEmulationScope.h"
#include <JavaScriptCore/InjectedScript.h>
#include <JavaScriptCore/InjectedScriptManager.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <JavaScriptCore/ScriptCallStackFactory.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageDebuggerAgent);

PageDebuggerAgent::PageDebuggerAgent(PageAgentContext& context)
    : WebDebuggerAgent(context)
    , m_inspectedPage(context.inspectedPage)
{
}

PageDebuggerAgent::~PageDebuggerAgent() = default;

bool PageDebuggerAgent::enabled() const
{
    return m_instrumentingAgents.enabledPageDebuggerAgent() == this && WebDebuggerAgent::enabled();
}

Inspector::Protocol::ErrorStringOr<std::tuple<Ref<Inspector::Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> PageDebuggerAgent::evaluateOnCallFrame(const Inspector::Protocol::Debugger::CallFrameId& callFrameId, const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture)
{
    auto injectedScript = injectedScriptManager().injectedScriptForObjectId(callFrameId);
    if (injectedScript.hasNoValue())
        return makeUnexpected("Missing injected script for given callFrameId"_s);

    UserGestureEmulationScope userGestureScope(m_inspectedPage, emulateUserGesture.value_or(false), dynamicDowncast<Document>(executionContext(injectedScript.globalObject())));
    return WebDebuggerAgent::evaluateOnCallFrame(injectedScript, callFrameId, expression, objectGroup, WTFMove(includeCommandLineAPI), WTFMove(doNotPauseOnExceptionsAndMuteConsole), WTFMove(returnByValue), WTFMove(generatePreview), WTFMove(saveResult), WTFMove(emulateUserGesture));
}

void PageDebuggerAgent::internalEnable()
{
    m_instrumentingAgents.setEnabledPageDebuggerAgent(this);

    WebDebuggerAgent::internalEnable();
}

void PageDebuggerAgent::internalDisable(bool isBeingDestroyed)
{
    m_instrumentingAgents.setEnabledPageDebuggerAgent(nullptr);

    WebDebuggerAgent::internalDisable(isBeingDestroyed);
}

String PageDebuggerAgent::sourceMapURLForScript(const JSC::Debugger::Script& script)
{
    static constexpr auto sourceMapHTTPHeader = "SourceMap"_s;
    static constexpr auto sourceMapHTTPHeaderDeprecated = "X-SourceMap"_s;

    if (!script.url.isEmpty()) {
        RefPtr localMainFrame = m_inspectedPage->localMainFrame();
        if (!localMainFrame)
            return String();

        CachedResource* resource = InspectorPageAgent::cachedResource(localMainFrame.get(), URL({ }, script.url));
        if (resource) {
            String sourceMapHeader = resource->response().httpHeaderField(StringView { sourceMapHTTPHeader });
            if (!sourceMapHeader.isEmpty())
                return sourceMapHeader;

            sourceMapHeader = resource->response().httpHeaderField(StringView { sourceMapHTTPHeaderDeprecated });
            if (!sourceMapHeader.isEmpty())
                return sourceMapHeader;
        }
    }

    return InspectorDebuggerAgent::sourceMapURLForScript(script);
}

void PageDebuggerAgent::muteConsole()
{
    PageConsoleClient::mute();
}

void PageDebuggerAgent::unmuteConsole()
{
    PageConsoleClient::unmute();
}

void PageDebuggerAgent::debuggerWillEvaluate(JSC::Debugger&, JSC::JSGlobalObject* globalObject, const JSC::Breakpoint::Action& action)
{
    m_breakpointActionUserGestureEmulationScopeStack.append(makeUniqueRef<UserGestureEmulationScope>(m_inspectedPage, action.emulateUserGesture, dynamicDowncast<Document>(executionContext(globalObject))));
}

void PageDebuggerAgent::debuggerDidEvaluate(JSC::Debugger&, JSC::JSGlobalObject*, const JSC::Breakpoint::Action&)
{
    m_breakpointActionUserGestureEmulationScopeStack.removeLast();
}

void PageDebuggerAgent::breakpointActionLog(JSC::JSGlobalObject* lexicalGlobalObject, const String& message)
{
    m_inspectedPage->console().addMessage(MessageSource::JS, MessageLevel::Log, message, createScriptCallStack(lexicalGlobalObject));
}

InjectedScript PageDebuggerAgent::injectedScriptForEval(Inspector::Protocol::ErrorString& errorString, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    RefPtr localMainFrame = m_inspectedPage->localMainFrame();
    if (!localMainFrame)
        return InjectedScript();

    if (!executionContextId)
        return injectedScriptManager().injectedScriptFor(&mainWorldGlobalObject(*localMainFrame));

    InjectedScript injectedScript = injectedScriptManager().injectedScriptForId(*executionContextId);
    if (injectedScript.hasNoValue())
        errorString = "Missing injected script for given executionContextId."_s;

    return injectedScript;
}

void PageDebuggerAgent::didClearWindowObjectInWorld(LocalFrame& frame, DOMWrapperWorld& world)
{
    if (!frame.isMainFrame() || &world != &mainThreadNormalWorld())
        return;

    didClearGlobalObject();
}

void PageDebuggerAgent::mainFrameStartedLoading()
{
    if (isPaused()) {
        setSuppressAllPauses(true);

        resume();
    }
}

void PageDebuggerAgent::mainFrameStoppedLoading()
{
    setSuppressAllPauses(false);
}

void PageDebuggerAgent::mainFrameNavigated()
{
    setSuppressAllPauses(false);
}

void PageDebuggerAgent::didRequestAnimationFrame(int callbackId, Document& document)
{
    if (!breakpointsActive())
        return;

    auto* globalObject = document.globalObject();
    if (!globalObject)
        return;

    didScheduleAsyncCall(globalObject, InspectorDebuggerAgent::AsyncCallType::RequestAnimationFrame, callbackId, true);
}

void PageDebuggerAgent::willFireAnimationFrame(int callbackId)
{
    willDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::RequestAnimationFrame, callbackId);
}

void PageDebuggerAgent::didCancelAnimationFrame(int callbackId)
{
    didCancelAsyncCall(InspectorDebuggerAgent::AsyncCallType::RequestAnimationFrame, callbackId);
}

void PageDebuggerAgent::didFireAnimationFrame(int callbackId)
{
    didDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::RequestAnimationFrame, callbackId);
}

} // namespace WebCore
