/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#include "WebInspectorInternal.h"

#include "WebFrame.h"
#include "WebInspectorMessages.h"
#include "WebInspectorUIMessages.h"
#include "WebInspectorUIProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/Chrome.h>
#include <WebCore/Document.h>
#include <WebCore/FrameLoadRequest.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/InspectorController.h>
#include <WebCore/InspectorFrontendClient.h>
#include <WebCore/InspectorPageAgent.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/NavigationAction.h>
#include <WebCore/NotImplemented.h>
#include <WebCore/Page.h>
#include <WebCore/ScriptController.h>
#include <WebCore/WindowFeatures.h>

static const float minimumAttachedHeight = 250;
static const float maximumAttachedHeightRatio = 0.75;
static const float minimumAttachedWidth = 500;

namespace WebKit {
using namespace WebCore;

Ref<WebInspector> WebInspector::create(WebPage& page)
{
    return adoptRef(*new WebInspector(page));
}

WebInspector::WebInspector(WebPage& page)
    : m_page(page)
{
}

WebInspector::~WebInspector()
{
    if (m_frontendConnection)
        m_frontendConnection->invalidate();
}

WebPage* WebInspector::page() const
{
    return m_page.get();
}

void WebInspector::openLocalInspectorFrontend()
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::RequestOpenLocalInspectorFrontend(), m_page->identifier());
}

void WebInspector::setFrontendConnection(IPC::Connection::Handle&& connectionHandle)
{
    // We might receive multiple updates if this web process got swapped into a WebPageProxy
    // shortly after another process established the connection.
    if (m_frontendConnection) {
        m_frontendConnection->invalidate();
        m_frontendConnection = nullptr;
    }

    if (!connectionHandle)
        return;

    m_frontendConnection = IPC::Connection::createClientConnection(IPC::Connection::Identifier { WTFMove(connectionHandle) });
    m_frontendConnection->open(*this);

    for (auto& callback : m_frontendConnectionActions)
        callback();
    m_frontendConnectionActions.clear();
}

void WebInspector::closeFrontendConnection()
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::DidClose(), m_page->identifier());

    // If we tried to close the frontend before it was created, then no connection exists yet.
    if (m_frontendConnection) {
        m_frontendConnection->invalidate();
        m_frontendConnection = nullptr;
    }

    m_frontendConnectionActions.clear();

    m_attached = false;
    m_previousCanAttach = false;
}

void WebInspector::bringToFront()
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::BringToFront(), m_page->identifier());
}

void WebInspector::whenFrontendConnectionEstablished(Function<void()>&& callback)
{
    if (m_frontendConnection) {
        callback();
        return;
    }

    m_frontendConnectionActions.append(WTFMove(callback));
}

// Called by WebInspector messages
void WebInspector::show(CompletionHandler<void()>&& completionHandler)
{
    if (!m_page->corePage())
        return;

    m_page->corePage()->inspectorController().show();
    completionHandler();
}

void WebInspector::close()
{
    if (!m_page->corePage())
        return;

    // Close could be called multiple times during teardown.
    if (!m_frontendConnection)
        return;

    closeFrontendConnection();
}

void WebInspector::evaluateScriptForTest(const String& script)
{
    if (!m_page->corePage())
        return;

    m_page->corePage()->inspectorController().evaluateForTestInFrontend(script);
}

void WebInspector::showConsole()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::ShowConsole(), 0);
    });
}

void WebInspector::showResources()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::ShowResources(), 0);
    });
}

void WebInspector::showMainResourceForFrame(WebCore::FrameIdentifier frameIdentifier)
{
    WebFrame* frame = WebProcess::singleton().webFrame(frameIdentifier);
    if (!frame)
        return;

    if (!m_page->corePage())
        return;

    String inspectorFrameIdentifier = m_page->corePage()->inspectorController().ensurePageAgent().frameId(frame->coreLocalFrame());

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::ShowMainResourceForFrame(inspectorFrameIdentifier), 0);
    });
}

void WebInspector::startPageProfiling()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::StartPageProfiling(), 0);
    });
}

void WebInspector::stopPageProfiling()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::StopPageProfiling(), 0);
    });
}

void WebInspector::startElementSelection()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::StartElementSelection(), 0);
    });
}

void WebInspector::stopElementSelection()
{
    if (!m_page->corePage())
        return;

    whenFrontendConnectionEstablished([=, this] {
        m_frontendConnection->send(Messages::WebInspectorUI::StopElementSelection(), 0);
    });
}

void WebInspector::elementSelectionChanged(bool active)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::ElementSelectionChanged(active), m_page->identifier());
}

void WebInspector::timelineRecordingChanged(bool active)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::TimelineRecordingChanged(active), m_page->identifier());
}

void WebInspector::setDeveloperPreferenceOverride(InspectorClient::DeveloperPreference developerPreference, std::optional<bool> overrideValue)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::SetDeveloperPreferenceOverride(developerPreference, overrideValue), m_page->identifier());
}

#if ENABLE(INSPECTOR_NETWORK_THROTTLING)

void WebInspector::setEmulatedConditions(std::optional<int64_t>&& bytesPerSecondLimit)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::SetEmulatedConditions(WTFMove(bytesPerSecondLimit)), m_page->identifier());
}

#endif // ENABLE(INSPECTOR_NETWORK_THROTTLING)

// FIXME <https://webkit.org/b/283435>: Remove this unused canAttachWindow function. Its return value is no longer used
// or respected by the UI process.
bool WebInspector::canAttachWindow()
{
    if (!m_page->corePage())
        return false;

    // Don't allow attaching to another inspector -- two inspectors in one window is too much!
    if (m_page->isInspectorPage())
        return false;

    // If we are already attached, allow attaching again to allow switching sides.
    if (m_attached)
        return true;

    // Don't allow the attach if the window would be too small to accommodate the minimum inspector size.
    RefPtr localMainFrame = m_page->localMainFrame();
    if (!localMainFrame)
        return false;
    unsigned inspectedPageHeight = localMainFrame->view()->visibleHeight();
    unsigned inspectedPageWidth = localMainFrame->view()->visibleWidth();
    unsigned maximumAttachedHeight = inspectedPageHeight * maximumAttachedHeightRatio;
    return minimumAttachedHeight <= maximumAttachedHeight && minimumAttachedWidth <= inspectedPageWidth;
}

void WebInspector::updateDockingAvailability()
{
    if (m_attached)
        return;

    bool canAttachWindow = this->canAttachWindow();
    if (m_previousCanAttach == canAttachWindow)
        return;

    m_previousCanAttach = canAttachWindow;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebInspectorUIProxy::AttachAvailabilityChanged(canAttachWindow), m_page->identifier());
}

} // namespace WebKit
