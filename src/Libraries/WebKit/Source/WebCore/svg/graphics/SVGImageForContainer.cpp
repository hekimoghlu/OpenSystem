/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#include "SVGImageForContainer.h"

#include "AffineTransform.h"
#include "FloatRect.h"
#include "FloatSize.h"
#include "Image.h"

namespace WebCore {

SVGImageForContainer::SVGImageForContainer(SVGImage* image, const FloatSize& containerSize, float containerZoom, const URL& initialFragmentURL)
    : m_image(image)
    , m_containerSize(containerSize)
    , m_containerZoom(containerZoom)
    , m_initialFragmentURL(initialFragmentURL)
{
}

RefPtr<SVGImage> SVGImageForContainer::protectedImage() const
{
    return m_image.get();
}

FloatSize SVGImageForContainer::size(ImageOrientation) const
{
    FloatSize scaledContainerSize(m_containerSize);
    scaledContainerSize.scale(m_containerZoom);
    return FloatSize(roundedIntSize(scaledContainerSize));
}

ImageDrawResult SVGImageForContainer::draw(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    return protectedImage()->drawForContainer(context, m_containerSize, m_containerZoom, m_initialFragmentURL, dstRect, srcRect, options);
}

void SVGImageForContainer::drawPattern(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, const AffineTransform& patternTransform,
    const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    protectedImage()->drawPatternForContainer(context, m_containerSize, m_containerZoom, m_initialFragmentURL, srcRect, patternTransform, phase, spacing, dstRect, options);
}

RefPtr<NativeImage> SVGImageForContainer::currentNativeImage()
{
    return protectedImage()->currentNativeImage();
}

} // namespace WebCore
