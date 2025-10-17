/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#import "config.h"
#import "TextTrackRepresentationCocoa.h"

#if ENABLE(VIDEO_PRESENTATION_MODE)

#import "VideoPresentationManager.h"
#import "VideoPresentationManagerProxyMessages.h"
#import "WebPage.h"
#import <WebCore/GraphicsContext.h>
#import <WebCore/HTMLVideoElement.h>
#import <WebCore/Page.h>

namespace WebKit {

WebTextTrackRepresentationCocoa::WebTextTrackRepresentationCocoa(WebCore::TextTrackRepresentationClient& client, WebCore::HTMLMediaElement& mediaElement)
    : WebCore::TextTrackRepresentationCocoa(client)
    , m_mediaElement(WeakPtr { mediaElement })
{
    auto* page = mediaElement.document().page();
    if (page)
        m_page = WeakPtr { WebPage::fromCorePage(*page) };
}

void WebTextTrackRepresentationCocoa::update()
{
    if (!m_page)
        return;
    Ref fullscreenManager = m_page->videoPresentationManager();
    if (!m_mediaElement || !is<WebCore::HTMLVideoElement>(m_mediaElement))
        return;
    
    auto image = m_client.createTextTrackRepresentationImage();
    if (!image)
        return;
    auto imageSize = image->size();
    RefPtr bitmap = WebCore::ShareableBitmap::create({ image->size(), image->colorSpace() });
    if (!bitmap)
        return;
    auto context = bitmap->createGraphicsContext();
    if (!context)
        return;
    context->drawNativeImage(*image, WebCore::FloatRect({ }, imageSize), WebCore::FloatRect({ }, imageSize), { WebCore::CompositeOperator::Copy });
    auto handle = bitmap->createHandle();
    if (!handle)
        return;
    Ref videoElement = downcast<WebCore::HTMLVideoElement>(*m_mediaElement);
    fullscreenManager->updateTextTrackRepresentationForVideoElement(videoElement, WTFMove(*handle));
}

void WebTextTrackRepresentationCocoa::setContentScale(float scale)
{
    WebCore::TextTrackRepresentationCocoa::setContentScale(scale);
    if (!m_page)
        return;
    Ref fullscreenManager = m_page->videoPresentationManager();
    RefPtr videoElement = dynamicDowncast<WebCore::HTMLVideoElement>(m_mediaElement.get());
    if (!videoElement)
        return;
    fullscreenManager->setTextTrackRepresentationContentScaleForVideoElement(*videoElement, scale);
}

void WebTextTrackRepresentationCocoa::setHidden(bool hidden) const
{
    WebCore::TextTrackRepresentationCocoa::setHidden(hidden);
    if (!m_page)
        return;
    Ref fullscreenManager = m_page->videoPresentationManager();
    RefPtr videoElement = dynamicDowncast<WebCore::HTMLVideoElement>(m_mediaElement.get());
    if (!videoElement)
        return;
    fullscreenManager->setTextTrackRepresentationIsHiddenForVideoElement(*videoElement, hidden);
}

void WebTextTrackRepresentationCocoa::setBounds(const WebCore::IntRect& bounds)
{
    if (m_bounds == bounds)
        return;
    m_bounds = bounds;
    client().textTrackRepresentationBoundsChanged(bounds);
}


} // namespace WebKit

#endif // ENABLE(VIDEO_PRESENTATION_MODE)
