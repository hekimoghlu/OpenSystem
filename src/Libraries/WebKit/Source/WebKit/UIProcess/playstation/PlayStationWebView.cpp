/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "PlayStationWebView.h"

#include "APIPageConfiguration.h"
#include "DrawingAreaProxyCoordinatedGraphics.h"
#include "WebProcessPool.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(WPE_BACKEND_PLAYSTATION)
#include <wpe/playstation.h>
#endif

namespace WebKit {

#if USE(WPE_BACKEND_PLAYSTATION)

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlayStationWebView);

RefPtr<PlayStationWebView> PlayStationWebView::create(struct wpe_view_backend* backend, const API::PageConfiguration& configuration)
{
    return adoptRef(*new PlayStationWebView(backend, configuration));
}

PlayStationWebView::PlayStationWebView(struct wpe_view_backend* backend, const API::PageConfiguration& conf)
    : m_pageClient(makeUniqueWithoutRefCountedCheck<PageClientImpl>(*this))
    , m_viewStateFlags { WebCore::ActivityState::WindowIsActive, WebCore::ActivityState::IsFocused, WebCore::ActivityState::IsVisible, WebCore::ActivityState::IsInWindow }
    , m_backend(backend)
{
    auto configuration = conf.copy();
    auto& pool = configuration->processPool();
    m_page = pool.createWebPage(*m_pageClient, WTFMove(configuration));

    wpe_view_backend_initialize(m_backend);

    auto& pageConfiguration = m_page->configuration();
    m_page->initializeWebPage(pageConfiguration.openedSite(), pageConfiguration.initialSandboxFlags());
}

#else

RefPtr<PlayStationWebView> PlayStationWebView::create(const API::PageConfiguration& configuration)
{
    return adoptRef(*new PlayStationWebView(configuration));
}

PlayStationWebView::PlayStationWebView(const API::PageConfiguration& conf)
    : m_pageClient(makeUniqueWithoutRefCountedCheck<PageClientImpl>(*this))
    , m_viewStateFlags { WebCore::ActivityState::WindowIsActive, WebCore::ActivityState::IsFocused, WebCore::ActivityState::IsVisible, WebCore::ActivityState::IsInWindow }
{
    auto configuration = conf.copy();
    auto& pool = configuration->processPool();
    m_page = pool.createWebPage(*m_pageClient, WTFMove(configuration));

    auto& pageConfiguration = m_page->configuration();
    m_page->initializeWebPage(pageConfiguration.openedSite(), pageConfiguration.initialSandboxFlags());
}

#endif // USE(WPE_BACKEND_PLAYSTATION)

PlayStationWebView::~PlayStationWebView()
{
}

void PlayStationWebView::setClient(std::unique_ptr<API::ViewClient>&& client)
{
    if (!client)
        m_client = makeUnique<API::ViewClient>();
    else
        m_client = WTFMove(client);
}

void PlayStationWebView::setViewSize(WebCore::IntSize viewSize)
{
    m_viewSize = viewSize;
}

void PlayStationWebView::setViewState(OptionSet<WebCore::ActivityState> flags)
{
    auto changedFlags = m_viewStateFlags ^ flags;
    m_viewStateFlags = flags;

    if (changedFlags)
        m_page->activityStateDidChange(changedFlags);
}

void PlayStationWebView::setViewNeedsDisplay(const WebCore::Region& region)
{
    if (m_client)
        m_client->setViewNeedsDisplay(*this, region);
}

#if ENABLE(FULLSCREEN_API)
void PlayStationWebView::willEnterFullScreen(CompletionHandler<void(bool)>&& completionHandler)
{
    m_isFullScreen = true;
    m_page->fullScreenManager()->willEnterFullScreen(WTFMove(completionHandler));
}

void PlayStationWebView::didEnterFullScreen()
{
    m_page->fullScreenManager()->didEnterFullScreen();
}

void PlayStationWebView::willExitFullScreen()
{
    m_page->fullScreenManager()->willExitFullScreen();
}

void PlayStationWebView::didExitFullScreen()
{
    m_page->fullScreenManager()->didExitFullScreen();
    m_isFullScreen = false;
}

void PlayStationWebView::requestExitFullScreen()
{
    if (isFullScreen())
        m_page->fullScreenManager()->requestExitFullScreen();
}

void PlayStationWebView::closeFullScreenManager()
{
    if (m_client && isFullScreen())
        m_client->closeFullScreen(*this);
    m_isFullScreen = false;
}

bool PlayStationWebView::isFullScreen()
{
    return m_isFullScreen;
}

void PlayStationWebView::enterFullScreen(CompletionHandler<void(bool)>&& completionHandler)
{
    if (m_client && !isFullScreen())
        m_client->enterFullScreen(*this, WTFMove(completionHandler));
    else
        completionHandler(false);
}

void PlayStationWebView::exitFullScreen()
{
    if (m_client && isFullScreen())
        m_client->exitFullScreen(*this);
}

void PlayStationWebView::beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
{
    if (m_client)
        m_client->beganEnterFullScreen(*this, initialFrame, finalFrame);
}

void PlayStationWebView::beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
{
    if (m_client)
        m_client->beganExitFullScreen(*this, initialFrame, finalFrame);
}
#endif

void PlayStationWebView::setCursor(const WebCore::Cursor& cursor)
{
    if (m_client)
        m_client->setCursor(*this, cursor);
}

} // namespace WebKit
