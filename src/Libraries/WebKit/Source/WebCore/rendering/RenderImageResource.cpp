/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#include "RenderImageResource.h"

#include "CachedImage.h"
#include "Image.h"
#include "RenderElement.h"
#include "RenderImage.h"
#include "RenderImageResourceStyleImage.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderImageResource);

RenderImageResource::RenderImageResource() = default;

RenderImageResource::~RenderImageResource() = default;

void RenderImageResource::initialize(RenderElement& renderer, CachedImage* styleCachedImage)
{
    ASSERT(!m_renderer);
    ASSERT(!m_cachedImage);
    m_renderer = renderer;
    m_cachedImage = styleCachedImage;
    m_cachedImageRemoveClientIsNeeded = !styleCachedImage;
}

void RenderImageResource::shutdown()
{
    image()->stopAnimation();
    setCachedImage(nullptr);
}

void RenderImageResource::setCachedImage(CachedResourceHandle<CachedImage>&& newImage)
{
    if (m_cachedImage == newImage)
        return;

    if (m_cachedImage && m_renderer && m_cachedImageRemoveClientIsNeeded)
        m_cachedImage->removeClient(*m_renderer);
    if (!m_renderer) {
        // removeClient may have destroyed the renderer.
        return;
    }
    m_cachedImage = WTFMove(newImage);
    m_cachedImageRemoveClientIsNeeded = true;
    if (!m_cachedImage)
        return;

    m_cachedImage->addClient(*renderer());
    if (m_cachedImage->errorOccurred())
        renderer()->imageChanged(m_cachedImage.get());
}

void RenderImageResource::resetAnimation()
{
    if (!m_cachedImage)
        return;

    image()->resetAnimation();

    if (m_renderer && !m_renderer->needsLayout())
        m_renderer->repaint();
}

RefPtr<Image> RenderImageResource::image(const IntSize&) const
{
    if (!m_cachedImage)
        return &Image::nullImage();
    if (auto image = m_cachedImage->imageForRenderer(m_renderer.get()))
        return image;
    return &Image::nullImage();
}

void RenderImageResource::setContainerContext(const IntSize& imageContainerSize, const URL& imageURL)
{
    if (!m_cachedImage || !m_renderer)
        return;
    m_cachedImage->setContainerContextForClient(*m_renderer, imageContainerSize, m_renderer->style().usedZoom(), imageURL);
}

LayoutSize RenderImageResource::imageSize(float multiplier, CachedImage::SizeType type) const
{
    if (!m_cachedImage)
        return LayoutSize();
    LayoutSize size = m_cachedImage->imageSizeForRenderer(m_renderer.get(), multiplier, type);
    if (auto* renderImage = dynamicDowncast<RenderImage>(m_renderer.get()))
        size.scale(renderImage->imageDevicePixelRatio());
    return size;
}

} // namespace WebCore
