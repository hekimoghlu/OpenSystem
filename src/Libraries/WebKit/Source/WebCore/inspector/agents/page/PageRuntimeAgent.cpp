/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include "PageRuntimeAgent.h"

#include "DOMWrapperWorld.h"
#include "Document.h"
#include "InspectorPageAgent.h"
#include "InstrumentingAgents.h"
#include "JSDOMWindowCustom.h"
#include "JSExecState.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include "ScriptController.h"
#include "SecurityOrigin.h"
#include "UserGestureEmulationScope.h"
#include <JavaScriptCore/InjectedScript.h>
#include <JavaScriptCore/InjectedScriptManager.h>
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageRuntimeAgent);

PageRuntimeAgent::PageRuntimeAgent(PageAgentContext& context)
    : InspectorRuntimeAgent(context)
    , m_frontendDispatcher(makeUnique<Inspector::RuntimeFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::RuntimeBackendDispatcher::create(Ref { context.backendDispatcher }, this))
    , m_instrumentingAgents(context.instrumentingAgents)
    , m_inspectedPage(context.inspectedPage)
{
}

PageRuntimeAgent::~PageRuntimeAgent() = default;

Inspector::Protocol::ErrorStringOr<void> PageRuntimeAgent::enable()
{
    if (m_instrumentingAgents.enabledPageRuntimeAgent() == this)
        return { };

    auto result = InspectorRuntimeAgent::enable();
    if (!result)
        return result;

    // Report initial contexts before enabling instrumentation as the reporting
    // can force creation of script state which could result in duplicate notifications.
    reportExecutionContextCreation();

    m_instrumentingAgents.setEnabledPageRuntimeAgent(this);

    return result;
}

Inspector::Protocol::ErrorStringOr<void> PageRuntimeAgent::disable()
{
    m_instrumentingAgents.setEnabledPageRuntimeAgent(nullptr);

    return InspectorRuntimeAgent::disable();
}

void PageRuntimeAgent::frameNavigated(LocalFrame& frame)
{
    // Ensure execution context is created for the frame even if it doesn't have scripts.
    mainWorldGlobalObject(frame);
}

void PageRuntimeAgent::didClearWindowObjectInWorld(LocalFrame& frame, DOMWrapperWorld& world)
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent)
        return;

    notifyContextCreated(pageAgent->frameId(&frame), frame.script().globalObject(world), world);
}

InjectedScript PageRuntimeAgent::injectedScriptForEval(Inspector::Protocol::ErrorString& errorString, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    if (!executionContextId) {
        RefPtr localMainFrame = m_inspectedPage->localMainFrame();
        if (!localMainFrame)
            return InjectedScript();

        InjectedScript result = injectedScriptManager().injectedScriptFor(&mainWorldGlobalObject(*localMainFrame));
        if (result.hasNoValue())
            errorString = "Internal error: main world execution context not found"_s;
        return result;
    }

    InjectedScript injectedScript = injectedScriptManager().injectedScriptForId(*executionContextId);
    if (injectedScript.hasNoValue())
        errorString = "Missing injected script for given executionContextId"_s;
    return injectedScript;
}

void PageRuntimeAgent::muteConsole()
{
    PageConsoleClient::mute();
}

void PageRuntimeAgent::unmuteConsole()
{
    PageConsoleClient::unmute();
}

void PageRuntimeAgent::reportExecutionContextCreation()
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent)
        return;

    m_inspectedPage->forEachLocalFrame([&](LocalFrame& frame) {
        if (!frame.script().canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript))
            return;

        auto frameId = pageAgent->frameId(&frame);

        // Always send the main world first.
        auto& mainGlobalObject = mainWorldGlobalObject(frame);
        notifyContextCreated(frameId, &mainGlobalObject, mainThreadNormalWorld());

        for (auto& jsWindowProxy : frame.windowProxy().jsWindowProxiesAsVector()) {
            auto* globalObject = jsWindowProxy->window();
            if (globalObject == &mainGlobalObject)
                continue;

            Ref securityOrigin = downcast<LocalDOMWindow>(jsWindowProxy->wrapped()).document()->securityOrigin();
            notifyContextCreated(frameId, globalObject, jsWindowProxy->protectedWorld(), securityOrigin.ptr());
        }
    });
}

static Inspector::Protocol::Runtime::ExecutionContextType toProtocol(DOMWrapperWorld::Type type)
{
    switch (type) {
    case DOMWrapperWorld::Type::Normal:
        return Inspector::Protocol::Runtime::ExecutionContextType::Normal;
    case DOMWrapperWorld::Type::User:
        return Inspector::Protocol::Runtime::ExecutionContextType::User;
    case DOMWrapperWorld::Type::Internal:
        return Inspector::Protocol::Runtime::ExecutionContextType::Internal;
    }

    ASSERT_NOT_REACHED();
    return Inspector::Protocol::Runtime::ExecutionContextType::Internal;
}

void PageRuntimeAgent::notifyContextCreated(const Inspector::Protocol::Network::FrameId& frameId, JSC::JSGlobalObject* globalObject, const DOMWrapperWorld& world, SecurityOrigin* securityOrigin)
{
    auto injectedScript = injectedScriptManager().injectedScriptFor(globalObject);
    if (injectedScript.hasNoValue())
        return;

    auto name = world.name();
    if (name.isEmpty() && securityOrigin)
        name = securityOrigin->toRawString();

    m_frontendDispatcher->executionContextCreated(Inspector::Protocol::Runtime::ExecutionContextDescription::create()
        .setId(injectedScriptManager().injectedScriptIdFor(globalObject))
        .setType(toProtocol(world.type()))
        .setName(name)
        .setFrameId(frameId)
        .release());
}

Inspector::Protocol::ErrorStringOr<std::tuple<Ref<Inspector::Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */, std::optional<int> /* savedResultIndex */>> PageRuntimeAgent::evaluate(const String& expression, const String& objectGroup, std::optional<bool>&& includeCommandLineAPI, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& saveResult, std::optional<bool>&& emulateUserGesture)
{
    Inspector::Protocol::ErrorString errorString;

    auto injectedScript = injectedScriptForEval(errorString, WTFMove(executionContextId));
    if (injectedScript.hasNoValue())
        return makeUnexpected(errorString);

    UserGestureEmulationScope userGestureScope(m_inspectedPage, emulateUserGesture.value_or(false), dynamicDowncast<Document>(executionContext(injectedScript.globalObject())));
    return InspectorRuntimeAgent::evaluate(injectedScript, expression, objectGroup, WTFMove(includeCommandLineAPI), WTFMove(doNotPauseOnExceptionsAndMuteConsole), WTFMove(returnByValue), WTFMove(generatePreview), WTFMove(saveResult), WTFMove(emulateUserGesture));
}

void PageRuntimeAgent::callFunctionOn(const Inspector::Protocol::Runtime::RemoteObjectId& objectId, const String& expression, RefPtr<JSON::Array>&& optionalArguments, std::optional<bool>&& doNotPauseOnExceptionsAndMuteConsole, std::optional<bool>&& returnByValue, std::optional<bool>&& generatePreview, std::optional<bool>&& emulateUserGesture, std::optional<bool>&& awaitPromise, Ref<CallFunctionOnCallback>&& callback)
{
    auto injectedScript = injectedScriptManager().injectedScriptForObjectId(objectId);
    if (injectedScript.hasNoValue()) {
        callback->sendFailure("Missing injected script for given objectId"_s);
        return;
    }

    UserGestureEmulationScope userGestureScope(m_inspectedPage, emulateUserGesture.value_or(false), dynamicDowncast<Document>(executionContext(injectedScript.globalObject())));
    return InspectorRuntimeAgent::callFunctionOn(objectId, expression, WTFMove(optionalArguments), WTFMove(doNotPauseOnExceptionsAndMuteConsole), WTFMove(returnByValue), WTFMove(generatePreview), WTFMove(emulateUserGesture), WTFMove(awaitPromise), WTFMove(callback));
}

} // namespace WebCore
