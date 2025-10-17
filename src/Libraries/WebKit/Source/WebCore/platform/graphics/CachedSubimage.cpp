/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
#include "CachedSubimage.h"

#include "GeometryUtilities.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CachedSubimage);

static FloatRect calculateCachedSubimageSourceRect(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, const FloatRect& imageRect)
{
    auto scaleFactor = destinationRect.size() / sourceRect.size();
    auto effectiveScaleFactor = scaleFactor * context.scaleFactor();

    auto cachedSubimageSourceSize = CachedSubimage::maxSide / effectiveScaleFactor;
    auto cachedSubimageSourceRect = FloatRect { sourceRect.center() - cachedSubimageSourceSize / 2, cachedSubimageSourceSize };

    auto shift = [](const FloatSize& delta) -> FloatSize {
        return FloatSize { std::max(0.0f, -delta.width()), std::max(0.0f, -delta.height()) };
    };

    // Move cachedSubimageSourceRect inside imageRect if needed.
    cachedSubimageSourceRect.move(shift(cachedSubimageSourceRect.location() - imageRect.location()));
    cachedSubimageSourceRect.move(-shift(imageRect.size() - cachedSubimageSourceRect.size()));
    cachedSubimageSourceRect.intersect(imageRect);

    return cachedSubimageSourceRect;
}

std::unique_ptr<CachedSubimage> CachedSubimage::create(GraphicsContext& context, const FloatSize& imageSize, const FloatRect& destinationRect, const FloatRect& sourceRect)
{
    auto cachedSubimageSourceRect = calculateCachedSubimageSourceRect(context, destinationRect, sourceRect, FloatRect { { }, imageSize });
    if (!(roundedIntRect(cachedSubimageSourceRect) == roundedIntRect(sourceRect) || cachedSubimageSourceRect.contains(sourceRect)))
        return nullptr;

    auto cachedSubimageDestinationRect = mapRect(cachedSubimageSourceRect, sourceRect, destinationRect);

    auto imageBuffer = context.createScaledImageBuffer(cachedSubimageDestinationRect, context.scaleFactor(), DestinationColorSpace::SRGB(), RenderingMode::Unaccelerated, RenderingMethod::Local);
    if (!imageBuffer)
        return nullptr;

    return makeUnique<CachedSubimage>(imageBuffer.releaseNonNull(), context.scaleFactor(), cachedSubimageDestinationRect, cachedSubimageSourceRect);
}

std::unique_ptr<CachedSubimage> CachedSubimage::createPixelated(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect)
{
    auto imageBuffer = context.createScaledImageBuffer(destinationRect, context.scaleFactor(), DestinationColorSpace::SRGB(), RenderingMode::Unaccelerated, RenderingMethod::Local);
    if (!imageBuffer)
        return nullptr;

    return makeUnique<CachedSubimage>(imageBuffer.releaseNonNull(), context.scaleFactor(), destinationRect, sourceRect);
}

CachedSubimage::CachedSubimage(Ref<ImageBuffer>&& imageBuffer, const FloatSize& scaleFactor, const FloatRect& destinationRect, const FloatRect& sourceRect)
    : m_imageBuffer(WTFMove(imageBuffer))
    , m_scaleFactor(scaleFactor)
    , m_destinationRect(destinationRect)
    , m_sourceRect(sourceRect)
{
}

bool CachedSubimage::canBeUsed(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect) const
{
    if (context.scaleFactor() != m_scaleFactor)
        return false;

    if (!areEssentiallyEqual(destinationRect.size() / sourceRect.size(), m_destinationRect.size() / m_sourceRect.size()))
        return false;

    return m_sourceRect.contains(sourceRect);
}

void CachedSubimage::draw(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect)
{
    ASSERT(canBeUsed(context, destinationRect, sourceRect));

    auto sourceRectScaled = sourceRect;
    sourceRectScaled.move(toFloatSize(-m_sourceRect.location()));

    auto scaleFactor = destinationRect.size() / sourceRect.size();
    sourceRectScaled.scale(scaleFactor * context.scaleFactor());

    context.drawImageBuffer(m_imageBuffer.get(), destinationRect, sourceRectScaled, { });
}

} // namespace WebCore
