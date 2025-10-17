/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include "SVGImageCache.h"

#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "LayoutSize.h"
#include "LegacyRenderSVGRoot.h"
#include "LocalFrameView.h"
#include "SVGImage.h"
#include "SVGImageForContainer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGImageCache);

SVGImageCache::SVGImageCache(SVGImage* svgImage)
    : m_svgImage(svgImage)
{
    ASSERT(m_svgImage);
}

SVGImageCache::~SVGImageCache()
{
    m_imageForContainerMap.clear();
}

void SVGImageCache::removeClientFromCache(const CachedImageClient* client)
{
    ASSERT(client);

    m_imageForContainerMap.remove(client);
}

void SVGImageCache::setContainerContextForClient(const CachedImageClient& client, const LayoutSize& containerSize, float containerZoom, const URL& imageURL)
{
    ASSERT(!containerSize.isEmpty());
    ASSERT(containerZoom);

    // SVG container has width or height less than 1 pixel.
    if (flooredIntSize(containerSize).isEmpty())
        return;

    FloatSize containerSizeWithoutZoom(containerSize);
    containerSizeWithoutZoom.scale(1 / containerZoom);

    m_imageForContainerMap.set(&client, SVGImageForContainer::create(protectedSVGImage().get(), containerSizeWithoutZoom, containerZoom, imageURL));
}

Image* SVGImageCache::findImageForRenderer(const RenderObject* renderer) const
{
    return renderer ? m_imageForContainerMap.get(renderer) : nullptr;
}

RefPtr<SVGImage> SVGImageCache::protectedSVGImage() const
{
    return m_svgImage.get();
}

FloatSize SVGImageCache::imageSizeForRenderer(const RenderObject* renderer) const
{
    SUPPRESS_UNCOUNTED_LOCAL auto* image = findImageForRenderer(renderer);
    return image ? image->size() : m_svgImage->size();
}

// FIXME: This doesn't take into account the animation timeline so animations will not
// restart on page load, nor will two animations in different pages have different timelines.
Image* SVGImageCache::imageForRenderer(const RenderObject* renderer) const
{
    auto* image = findImageForRenderer(renderer);
    if (!image)
        return &Image::nullImage();
    ASSERT(!image->size().isEmpty());
    return image;
}

} // namespace WebCore
