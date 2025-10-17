/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#import "WebMediaPlaybackTargetPicker.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#import "WebView.h"
#import <WebCore/MediaPlaybackTarget.h>
#import <WebCore/Page.h>
#import <WebCore/WebMediaSessionManager.h>
#import <wtf/TZoneMallocInlines.h>

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebMediaPlaybackTargetPicker);

std::unique_ptr<WebMediaPlaybackTargetPicker> WebMediaPlaybackTargetPicker::create(WebView *webView, WebCore::Page& page)
{
    return makeUnique<WebMediaPlaybackTargetPicker>(webView, page);
}

WebMediaPlaybackTargetPicker::WebMediaPlaybackTargetPicker(WebView *webView, WebCore::Page& page)
    : m_page(page)
    , m_webView(webView)
{
}

void WebMediaPlaybackTargetPicker::addPlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier contextId)
{
    WebCore::WebMediaSessionManager::shared().addPlaybackTargetPickerClient(*this, contextId);
}

void WebMediaPlaybackTargetPicker::removePlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier contextId)
{
    WebCore::WebMediaSessionManager::shared().removePlaybackTargetPickerClient(*this, contextId);
}

void WebMediaPlaybackTargetPicker::showPlaybackTargetPicker(WebCore::PlaybackTargetClientContextIdentifier contextId, const WebCore::FloatRect& rect, bool hasVideo)
{
    WebCore::WebMediaSessionManager::shared().showPlaybackTargetPicker(*this, contextId, WebCore::IntRect(rect), hasVideo, m_page ? m_page->useDarkAppearance() : false);
}

void WebMediaPlaybackTargetPicker::playbackTargetPickerClientStateDidChange(WebCore::PlaybackTargetClientContextIdentifier contextId, WebCore::MediaProducerMediaStateFlags state)
{
    WebCore::WebMediaSessionManager::shared().clientStateDidChange(*this, contextId, state);
}

void WebMediaPlaybackTargetPicker::setMockMediaPlaybackTargetPickerEnabled(bool enabled)
{
    WebCore::WebMediaSessionManager::shared().setMockMediaPlaybackTargetPickerEnabled(enabled);
}

void WebMediaPlaybackTargetPicker::setMockMediaPlaybackTargetPickerState(const String& name, WebCore::MediaPlaybackTargetContext::MockState state)
{
    WebCore::WebMediaSessionManager::shared().setMockMediaPlaybackTargetPickerState(name, state);
}

void WebMediaPlaybackTargetPicker::mockMediaPlaybackTargetPickerDismissPopup()
{
    WebCore::WebMediaSessionManager::shared().mockMediaPlaybackTargetPickerDismissPopup();
}

void WebMediaPlaybackTargetPicker::setPlaybackTarget(WebCore::PlaybackTargetClientContextIdentifier contextId, Ref<WebCore::MediaPlaybackTarget>&& target)
{
    if (!m_page)
        return;

    m_page->setPlaybackTarget(contextId, WTFMove(target));
}

void WebMediaPlaybackTargetPicker::externalOutputDeviceAvailableDidChange(WebCore::PlaybackTargetClientContextIdentifier contextId, bool available)
{
    if (!m_page)
        return;

    m_page->playbackTargetAvailabilityDidChange(contextId, available);
}

void WebMediaPlaybackTargetPicker::setShouldPlayToPlaybackTarget(WebCore::PlaybackTargetClientContextIdentifier contextId, bool shouldPlay)
{
    if (!m_page)
        return;

    m_page->setShouldPlayToPlaybackTarget(contextId, shouldPlay);
}

void WebMediaPlaybackTargetPicker::playbackTargetPickerWasDismissed(WebCore::PlaybackTargetClientContextIdentifier contextId)
{
    if (m_page)
        m_page->playbackTargetPickerWasDismissed(contextId);
}

void WebMediaPlaybackTargetPicker::invalidate()
{
    m_page = nullptr;
    m_webView = nil;
    WebCore::WebMediaSessionManager::shared().removeAllPlaybackTargetPickerClients(*this);
}

RetainPtr<PlatformView> WebMediaPlaybackTargetPicker::platformView() const
{
    ASSERT(m_webView);
    return m_webView.get();
}

#endif
