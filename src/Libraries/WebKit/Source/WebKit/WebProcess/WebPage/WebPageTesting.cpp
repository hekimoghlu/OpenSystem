/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#include "WebPageTesting.h"

#include "DrawingArea.h"
#include "NotificationPermissionRequestManager.h"
#include "PluginView.h"
#include "WebBackForwardListProxy.h"
#include "WebNotificationClient.h"
#include "WebPage.h"
#include "WebPageTestingMessages.h"
#include "WebProcess.h"
#include <WebCore/BackForwardController.h>
#include <WebCore/Editor.h>
#include <WebCore/FocusController.h>
#include <WebCore/IntPoint.h>
#include <WebCore/NotificationController.h>
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageTesting);

Ref<WebPageTesting> WebPageTesting::create(WebPage& page)
{
    return adoptRef(*new WebPageTesting(page));
}

WebPageTesting::WebPageTesting(WebPage& page)
    : m_page(page)
    , m_pageIdentifier(page.identifier())
{
    WebProcess::singleton().addMessageReceiver(Messages::WebPageTesting::messageReceiverName(), m_pageIdentifier, *this);
}

WebPageTesting::~WebPageTesting()
{
    WebProcess::singleton().removeMessageReceiver(Messages::WebPageTesting::messageReceiverName(), m_pageIdentifier);
}

void WebPageTesting::isLayerTreeFrozen(CompletionHandler<void(bool)>&& completionHandler)
{
    completionHandler(m_page && !!m_page->layerTreeFreezeReasons());
}

void WebPageTesting::setPermissionLevel(const String& origin, bool allowed)
{
#if ENABLE(NOTIFICATIONS)
    RefPtr page = m_page.get();
    if (RefPtr notificationPermissionRequestManager = page ? page->notificationPermissionRequestManager() : nullptr)
        notificationPermissionRequestManager->setPermissionLevelForTesting(origin, allowed);
#else
    UNUSED_PARAM(origin);
    UNUSED_PARAM(allowed);
#endif
}

void WebPageTesting::isEditingCommandEnabled(const String& commandName, CompletionHandler<void(bool)>&& completionHandler)
{
    RefPtr page = m_page.get();
    if (!page)
        return completionHandler(false);

    RefPtr corePage = page->corePage();
    RefPtr frame = corePage->checkedFocusController()->focusedOrMainFrame();
    if (!frame)
        return completionHandler(false);

#if ENABLE(PDF_PLUGIN)
    if (RefPtr pluginView = page->focusedPluginViewForFrame(*frame))
        return completionHandler(pluginView->isEditingCommandEnabled(commandName));
#endif

    auto command = frame->protectedEditor()->command(commandName);
    completionHandler(command.isSupported() && command.isEnabled());
}

#if ENABLE(NOTIFICATIONS)
void WebPageTesting::clearNotificationPermissionState()
{
    RefPtr page = m_page ? m_page->corePage() : nullptr;
    auto& client = static_cast<WebNotificationClient&>(WebCore::NotificationController::from(page.get())->client());
    client.clearNotificationPermissionState();
}
#endif

void WebPageTesting::clearWheelEventTestMonitor()
{
    RefPtr page = m_page ? m_page->corePage() : nullptr;
    if (!page)
        return;

    page->clearWheelEventTestMonitor();
}

void WebPageTesting::setTopContentInset(float contentInset, CompletionHandler<void()>&& completionHandler)
{
    if (RefPtr page = m_page.get())
        page->setTopContentInset(contentInset);
    completionHandler();
}

void WebPageTesting::resetStateBetweenTests()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    if (RefPtr mainFrame = page->mainFrame()) {
        mainFrame->disownOpener();
        mainFrame->tree().clearName();
    }
    if (RefPtr corePage = page->corePage()) {
        // Force consistent "responsive" behavior for WebPage::eventThrottlingDelay() for testing. Tests can override via internals.
        corePage->setEventThrottlingBehaviorOverride(WebCore::EventThrottlingBehavior::Responsive);
    }
}

void WebPageTesting::clearCachedBackForwardListCounts(CompletionHandler<void()>&& completionHandler)
{
    RefPtr page = m_page ? m_page->corePage() : nullptr;
    if (!page)
        return completionHandler();

    Ref backForwardListProxy = static_cast<WebBackForwardListProxy&>(page->backForward().client());
    backForwardListProxy->clearCachedListCounts();
    completionHandler();
}

void WebPageTesting::setTracksRepaints(bool trackRepaints, CompletionHandler<void()>&& completionHandler)
{
    RefPtr page = m_page ? m_page->corePage() : nullptr;
    if (!page)
        return completionHandler();

    for (auto& rootFrame : page->rootFrames()) {
        if (RefPtr view = rootFrame->view())
            view->setTracksRepaints(trackRepaints);
    }
    completionHandler();
}

void WebPageTesting::displayAndTrackRepaints(CompletionHandler<void()>&& completionHandler)
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    RefPtr corePage = m_page->corePage();
    if (!corePage)
        return completionHandler();

    page->protectedDrawingArea()->updateRenderingWithForcedRepaint();
    for (auto& rootFrame : corePage->rootFrames()) {
        if (RefPtr view = rootFrame->view()) {
            view->setTracksRepaints(true);
            view->resetTrackedRepaints();
        }
    }
    completionHandler();
}

} // namespace WebKit
