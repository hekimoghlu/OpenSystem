/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#include "WPEWebView.h"

#include "APIPageConfiguration.h"
#include "APIViewClient.h"
#include "DrawingAreaProxy.h"
#include "EditingRange.h"
#include "EditorState.h"
#include "WebPreferences.h"
#include "WebProcessPool.h"
#include <WebCore/CompositionUnderline.h>

using namespace WebKit;

namespace WKWPE {

View::View()
    : m_client(makeUnique<API::ViewClient>())
    , m_pageClient(makeUniqueWithoutRefCountedCheck<PageClientImpl>(*this))
{
}

View::~View()
{
#if USE(ATK)
    if (m_accessible)
        webkitWebViewAccessibleSetWebView(m_accessible.get(), nullptr);
#endif
    m_pageProxy->close();
}

void View::createWebPage(const API::PageConfiguration& configuration)
{
    auto& pool = configuration.processPool();
    m_pageProxy = pool.createWebPage(*m_pageClient, configuration.copy());

#if ENABLE(MEMORY_SAMPLER)
    if (getenv("WEBKIT_SAMPLE_MEMORY"))
        pool.startMemorySampler(0);
#endif
}

void View::setClient(std::unique_ptr<API::ViewClient>&& client)
{
    if (!client)
        m_client = makeUnique<API::ViewClient>();
    else
        m_client = WTFMove(client);
}

void View::frameDisplayed()
{
    m_client->frameDisplayed(*this);
}

void View::willStartLoad()
{
    m_client->willStartLoad(*this);
}

void View::didChangePageID()
{
    m_client->didChangePageID(*this);
}

void View::didReceiveUserMessage(UserMessage&& message, CompletionHandler<void(UserMessage&&)>&& completionHandler)
{
    m_client->didReceiveUserMessage(*this, WTFMove(message), WTFMove(completionHandler));
}

WebKitWebResourceLoadManager* View::webResourceLoadManager()
{
    return m_client->webResourceLoadManager();
}

void View::setInputMethodContext(WebKitInputMethodContext* context)
{
    m_inputMethodFilter.setContext(context);
}

WebKitInputMethodContext* View::inputMethodContext() const
{
    return m_inputMethodFilter.context();
}

void View::setInputMethodState(std::optional<InputMethodState>&& state)
{
    m_inputMethodFilter.setState(WTFMove(state));
}

void View::selectionDidChange()
{
    const auto& editorState = m_pageProxy->editorState();
    if (editorState.hasPostLayoutAndVisualData()) {
        m_inputMethodFilter.notifyCursorRect(editorState.visualData->caretRectAtStart);
        m_inputMethodFilter.notifySurrounding(editorState.postLayoutData->surroundingContext, editorState.postLayoutData->surroundingContextCursorPosition,
            editorState.postLayoutData->surroundingContextSelectionPosition);
    }
}

void View::setSize(const WebCore::IntSize& size)
{
    m_size = size;
    if (m_pageProxy->drawingArea())
        m_pageProxy->drawingArea()->setSize(size);
}

void View::close()
{
    m_pageProxy->close();
}

#if USE(ATK)
WebKitWebViewAccessible* View::accessible() const
{
    if (!m_accessible)
        m_accessible = webkitWebViewAccessibleNew(const_cast<View*>(this));
    return m_accessible.get();
}
#endif

#if ENABLE(FULLSCREEN_API)
bool View::isFullScreen() const
{
    return m_fullscreenState == WebFullScreenManagerProxy::FullscreenState::EnteringFullscreen || m_fullscreenState == WebFullScreenManagerProxy::FullscreenState::InFullscreen;
}

void View::willEnterFullScreen(CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(m_fullscreenState == WebFullScreenManagerProxy::FullscreenState::NotInFullscreen);
    if (auto* fullScreenManagerProxy = page().fullScreenManager())
        fullScreenManagerProxy->willEnterFullScreen(WTFMove(completionHandler));
    else
        completionHandler(false);
    m_fullscreenState = WebFullScreenManagerProxy::FullscreenState::EnteringFullscreen;
}

void View::willExitFullScreen()
{
    ASSERT(m_fullscreenState == WebFullScreenManagerProxy::FullscreenState::EnteringFullscreen || m_fullscreenState == WebFullScreenManagerProxy::FullscreenState::InFullscreen);

    if (auto* fullScreenManagerProxy = page().fullScreenManager())
        fullScreenManagerProxy->willExitFullScreen();
    m_fullscreenState = WebFullScreenManagerProxy::FullscreenState::ExitingFullscreen;
}
#endif // ENABLE(FULLSCREEN_API)

} // namespace WKWPE
