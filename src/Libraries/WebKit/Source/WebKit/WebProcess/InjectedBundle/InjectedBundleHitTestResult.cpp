/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#include "InjectedBundleHitTestResult.h"

#include "InjectedBundleNodeHandle.h"
#include "WebFrame.h"
#include "WebImage.h"
#include "WebLocalFrameLoaderClient.h"
#include <WebCore/BitmapImage.h>
#include <WebCore/Document.h>
#include <WebCore/Element.h>
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/HTMLMediaElement.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <wtf/URL.h>

namespace WebKit {
using namespace WebCore;

Ref<InjectedBundleHitTestResult> InjectedBundleHitTestResult::create(const HitTestResult& hitTestResult)
{
    return adoptRef(*new InjectedBundleHitTestResult(hitTestResult));
}

RefPtr<InjectedBundleNodeHandle> InjectedBundleHitTestResult::nodeHandle() const
{
    return InjectedBundleNodeHandle::getOrCreate(m_hitTestResult.innerNonSharedNode());
}

RefPtr<InjectedBundleNodeHandle> InjectedBundleHitTestResult::urlElementHandle() const
{
    return InjectedBundleNodeHandle::getOrCreate(m_hitTestResult.URLElement());
}

RefPtr<WebFrame> InjectedBundleHitTestResult::frame() const
{
    auto* node = m_hitTestResult.innerNonSharedNode();
    if (!node)
        return nullptr;

    auto* frame = node->document().frame();
    if (!frame)
        return nullptr;

    return WebFrame::fromCoreFrame(*frame);
}

RefPtr<WebFrame> InjectedBundleHitTestResult::targetFrame() const
{
    auto* frame = m_hitTestResult.targetFrame();
    if (!frame)
        return nullptr;

    return WebFrame::fromCoreFrame(*frame);
}

String InjectedBundleHitTestResult::absoluteImageURL() const
{
    return m_hitTestResult.absoluteImageURL().string();
}

String InjectedBundleHitTestResult::absolutePDFURL() const
{
    return m_hitTestResult.absolutePDFURL().string();
}

String InjectedBundleHitTestResult::absoluteLinkURL() const
{
    return m_hitTestResult.absoluteLinkURL().string();
}

String InjectedBundleHitTestResult::absoluteMediaURL() const
{
    return m_hitTestResult.absoluteMediaURL().string();
}

bool InjectedBundleHitTestResult::mediaIsInFullscreen() const
{
    return m_hitTestResult.mediaIsInFullscreen();
}

bool InjectedBundleHitTestResult::mediaHasAudio() const
{
    return m_hitTestResult.mediaHasAudio();
}

bool InjectedBundleHitTestResult::isDownloadableMedia() const
{
    return m_hitTestResult.isDownloadableMedia();
}

BundleHitTestResultMediaType InjectedBundleHitTestResult::mediaType() const
{
#if !ENABLE(VIDEO)
    return BundleHitTestResultMediaTypeNone;
#else
    if (!is<HTMLMediaElement>(m_hitTestResult.innerNonSharedNode()))
        return BundleHitTestResultMediaTypeNone;
    return m_hitTestResult.mediaIsVideo() ? BundleHitTestResultMediaTypeVideo : BundleHitTestResultMediaTypeAudio;
#endif
}

String InjectedBundleHitTestResult::linkLabel() const
{
    return m_hitTestResult.textContent();
}

String InjectedBundleHitTestResult::linkTitle() const
{
    return m_hitTestResult.titleDisplayString();
}

String InjectedBundleHitTestResult::linkSuggestedFilename() const
{
    return m_hitTestResult.linkSuggestedFilename();
}

IntRect InjectedBundleHitTestResult::imageRect() const
{
    IntRect imageRect = m_hitTestResult.imageRect();
    if (imageRect.isEmpty())
        return imageRect;
        
    // The image rect in HitTestResult is in frame coordinates, but we need it in WKView
    // coordinates since WebKit2 clients don't have enough context to do the conversion themselves.
    auto webFrame = frame();
    if (!webFrame)
        return imageRect;
    
    auto* coreFrame = webFrame->coreLocalFrame();
    if (!coreFrame)
        return imageRect;
    
    auto* view = coreFrame->view();
    if (!view)
        return imageRect;
    
    return view->contentsToRootView(imageRect);
}

RefPtr<WebImage> InjectedBundleHitTestResult::image() const
{
    // For now, we only handle bitmap images.
    auto* bitmapImage = dynamicDowncast<BitmapImage>(m_hitTestResult.image());
    if (!bitmapImage)
        return nullptr;

    IntSize size(bitmapImage->size());
    auto webImage = WebImage::create(size, { }, DestinationColorSpace::SRGB());
    if (!webImage->context())
        return nullptr;

    // FIXME: need to handle EXIF rotation.
    auto& graphicsContext = *webImage->context();
    graphicsContext.drawImage(*bitmapImage, { { }, size });

    return webImage;
}

bool InjectedBundleHitTestResult::isSelected() const
{
    return m_hitTestResult.isSelected();
}

} // WebKit
