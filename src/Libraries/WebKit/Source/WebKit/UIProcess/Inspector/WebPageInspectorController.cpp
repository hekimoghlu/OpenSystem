/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#include "WebPageInspectorController.h"

#include "APIUIClient.h"
#include "InspectorBrowserAgent.h"
#include "ProvisionalPageProxy.h"
#include "WebFrameProxy.h"
#include "WebPageInspectorAgentBase.h"
#include "WebPageInspectorTarget.h"
#include "WebPageProxy.h"
#include <JavaScriptCore/InspectorAgentBase.h>
#include <JavaScriptCore/InspectorBackendDispatcher.h>
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendRouter.h>
#include <JavaScriptCore/InspectorTargetAgent.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace Inspector;

static String getTargetID(const ProvisionalPageProxy& provisionalPage)
{
    return WebPageInspectorTarget::toTargetID(provisionalPage.webPageID());
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageInspectorController);

WebPageInspectorController::WebPageInspectorController(WebPageProxy& inspectedPage)
    : m_frontendRouter(FrontendRouter::create())
    , m_backendDispatcher(BackendDispatcher::create(m_frontendRouter.copyRef()))
    , m_inspectedPage(inspectedPage)
{
    auto targetAgent = makeUnique<InspectorTargetAgent>(m_frontendRouter.get(), m_backendDispatcher.get());
    m_targetAgent = targetAgent.get();
    m_agents.append(WTFMove(targetAgent));
}

WebPageInspectorController::~WebPageInspectorController() = default;

Ref<WebPageProxy> WebPageInspectorController::protectedInspectedPage()
{
    return m_inspectedPage.get();
}

void WebPageInspectorController::init()
{
    String pageTargetId = WebPageInspectorTarget::toTargetID(m_inspectedPage->webPageIDInMainFrameProcess());
    createInspectorTarget(pageTargetId, Inspector::InspectorTargetType::Page);
}

void WebPageInspectorController::pageClosed()
{
    disconnectAllFrontends();

    m_agents.discardValues();
}

bool WebPageInspectorController::hasLocalFrontend() const
{
    return m_frontendRouter->hasLocalFrontend();
}

void WebPageInspectorController::connectFrontend(Inspector::FrontendChannel& frontendChannel, bool, bool)
{
    createLazyAgents();

    bool connectingFirstFrontend = !m_frontendRouter->hasFrontends();

    m_frontendRouter->connectFrontend(frontendChannel);

    if (connectingFirstFrontend)
        m_agents.didCreateFrontendAndBackend(&m_frontendRouter.get(), &m_backendDispatcher.get());

    auto inspectedPage = protectedInspectedPage();
    inspectedPage->didChangeInspectorFrontendCount(m_frontendRouter->frontendCount());

#if ENABLE(REMOTE_INSPECTOR)
    if (hasLocalFrontend())
        inspectedPage->remoteInspectorInformationDidChange();
#endif
}

void WebPageInspectorController::disconnectFrontend(FrontendChannel& frontendChannel)
{
    m_frontendRouter->disconnectFrontend(frontendChannel);

    bool disconnectingLastFrontend = !m_frontendRouter->hasFrontends();
    if (disconnectingLastFrontend)
        m_agents.willDestroyFrontendAndBackend(DisconnectReason::InspectorDestroyed);

    auto inspectedPage = protectedInspectedPage();
    inspectedPage->didChangeInspectorFrontendCount(m_frontendRouter->frontendCount());

#if ENABLE(REMOTE_INSPECTOR)
    if (disconnectingLastFrontend)
        inspectedPage->remoteInspectorInformationDidChange();
#endif
}

void WebPageInspectorController::disconnectAllFrontends()
{
    // FIXME: Handle a local inspector client.

    if (!m_frontendRouter->hasFrontends())
        return;

    // Notify agents first, since they may need to use InspectorClient.
    m_agents.willDestroyFrontendAndBackend(DisconnectReason::InspectedTargetDestroyed);

    // Disconnect any remaining remote frontends.
    m_frontendRouter->disconnectAllFrontends();

    auto inspectedPage = protectedInspectedPage();
    inspectedPage->didChangeInspectorFrontendCount(m_frontendRouter->frontendCount());

#if ENABLE(REMOTE_INSPECTOR)
    inspectedPage->remoteInspectorInformationDidChange();
#endif
}

void WebPageInspectorController::dispatchMessageFromFrontend(const String& message)
{
    m_backendDispatcher->dispatch(message);
}

#if ENABLE(REMOTE_INSPECTOR)
void WebPageInspectorController::setIndicating(bool indicating)
{
    auto inspectedPage = protectedInspectedPage();
#if !PLATFORM(IOS_FAMILY)
    inspectedPage->setIndicating(indicating);
#else
    if (indicating)
        inspectedPage->showInspectorIndication();
    else
        inspectedPage->hideInspectorIndication();
#endif
}
#endif

void WebPageInspectorController::createInspectorTarget(const String& targetId, Inspector::InspectorTargetType type)
{
    addTarget(InspectorTargetProxy::create(protectedInspectedPage(), targetId, type));
}

void WebPageInspectorController::destroyInspectorTarget(const String& targetId)
{
    auto it = m_targets.find(targetId);
    if (it == m_targets.end())
        return;
    m_targetAgent->targetDestroyed(*it->value);
    m_targets.remove(it);
}

void WebPageInspectorController::sendMessageToInspectorFrontend(const String& targetId, const String& message)
{
    m_targetAgent->sendMessageFromTargetToFrontend(targetId, message);
}

bool WebPageInspectorController::shouldPauseLoading(const ProvisionalPageProxy& provisionalPage) const
{
    if (!m_frontendRouter->hasFrontends())
        return false;

    auto* target = m_targets.get(getTargetID(provisionalPage));
    ASSERT(target);
    return target->isPaused();
}

void WebPageInspectorController::setContinueLoadingCallback(const ProvisionalPageProxy& provisionalPage, WTF::Function<void()>&& callback)
{
    auto* target = m_targets.get(getTargetID(provisionalPage));
    ASSERT(target);
    target->setResumeCallback(WTFMove(callback));
}

void WebPageInspectorController::didCreateProvisionalPage(ProvisionalPageProxy& provisionalPage)
{
    addTarget(InspectorTargetProxy::create(provisionalPage, getTargetID(provisionalPage), Inspector::InspectorTargetType::Page));
}

void WebPageInspectorController::willDestroyProvisionalPage(const ProvisionalPageProxy& provisionalPage)
{
    destroyInspectorTarget(getTargetID(provisionalPage));
}

void WebPageInspectorController::didCommitProvisionalPage(WebCore::PageIdentifier oldWebPageID, WebCore::PageIdentifier newWebPageID)
{
    String oldID = WebPageInspectorTarget::toTargetID(oldWebPageID);
    String newID = WebPageInspectorTarget::toTargetID(newWebPageID);
    auto newTarget = m_targets.take(newID);
    ASSERT(newTarget);
    newTarget->didCommitProvisionalTarget();
    m_targetAgent->didCommitProvisionalTarget(oldID, newID);

    // We've disconnected from the old page and will not receive any message from it, so
    // we destroy everything but the new target here.
    // FIXME: <https://webkit.org/b/202937> do not destroy targets that belong to the committed page.
    for (auto& target : m_targets.values())
        m_targetAgent->targetDestroyed(*target);
    m_targets.clear();
    m_targets.set(newTarget->identifier(), WTFMove(newTarget));
}

InspectorBrowserAgent* WebPageInspectorController::enabledBrowserAgent() const
{
    return m_enabledBrowserAgent.get();
}

WebPageAgentContext WebPageInspectorController::webPageAgentContext()
{
    return {
        m_frontendRouter.get(),
        m_backendDispatcher.get(),
        m_inspectedPage,
    };
}

void WebPageInspectorController::createLazyAgents()
{
    if (m_didCreateLazyAgents)
        return;

    m_didCreateLazyAgents = true;

    auto webPageContext = webPageAgentContext();

    m_agents.append(makeUnique<InspectorBrowserAgent>(webPageContext));
}

void WebPageInspectorController::addTarget(std::unique_ptr<InspectorTargetProxy>&& target)
{
    m_targetAgent->targetCreated(*target);
    m_targets.set(target->identifier(), WTFMove(target));
}

void WebPageInspectorController::setEnabledBrowserAgent(InspectorBrowserAgent* agent)
{
    if (m_enabledBrowserAgent == agent)
        return;

    m_enabledBrowserAgent = agent;

    auto inspectedPage = protectedInspectedPage();
    if (m_enabledBrowserAgent)
        inspectedPage->uiClient().didEnableInspectorBrowserDomain(inspectedPage);
    else
        inspectedPage->uiClient().didDisableInspectorBrowserDomain(inspectedPage);
}

void WebPageInspectorController::browserExtensionsEnabled(HashMap<String, String>&& extensionIDToName)
{
    if (m_enabledBrowserAgent)
        m_enabledBrowserAgent->extensionsEnabled(WTFMove(extensionIDToName));
}

void WebPageInspectorController::browserExtensionsDisabled(HashSet<String>&& extensionIDs)
{
    if (m_enabledBrowserAgent)
        m_enabledBrowserAgent->extensionsDisabled(WTFMove(extensionIDs));
}

} // namespace WebKit
