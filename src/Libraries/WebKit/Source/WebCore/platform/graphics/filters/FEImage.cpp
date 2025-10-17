/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
#include "FEImage.h"

#include "FEImageSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEImage> FEImage::create(SourceImage&& sourceImage, const FloatRect& sourceImageRect, const SVGPreserveAspectRatioValue& preserveAspectRatio)
{
    return adoptRef(*new FEImage(WTFMove(sourceImage), sourceImageRect, preserveAspectRatio));
}

FEImage::FEImage(SourceImage&& sourceImage, const FloatRect& sourceImageRect, const SVGPreserveAspectRatioValue& preserveAspectRatio)
    : FilterEffect(Type::FEImage)
    , m_sourceImage(WTFMove(sourceImage))
    , m_sourceImageRect(sourceImageRect)
    , m_preserveAspectRatio(preserveAspectRatio)
{
}

bool FEImage::operator==(const FEImage& other) const
{
    return FilterEffect::operator==(other)
        && m_sourceImage == other.m_sourceImage
        && m_sourceImageRect == other.m_sourceImageRect
        && m_preserveAspectRatio == other.m_preserveAspectRatio;
}

FloatRect FEImage::calculateImageRect(const Filter& filter, std::span<const FloatRect>, const FloatRect& primitiveSubregion) const
{
    if (m_sourceImage.nativeImageIfExists()) {
        auto imageRect = primitiveSubregion;
        auto srcRect = m_sourceImageRect;
        m_preserveAspectRatio.transformRect(imageRect, srcRect);
        return filter.clipToMaxEffectRect(imageRect, primitiveSubregion);
    }

    if (m_sourceImage.imageBufferIfExists())
        return filter.maxEffectRect(primitiveSubregion);

    ASSERT_NOT_REACHED();
    return FloatRect();
}

std::unique_ptr<FilterEffectApplier> FEImage::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEImageSoftwareApplier>(*this);
}

TextStream& FEImage::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feImage";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " image-size=\"" << m_sourceImageRect.width() << "x" << m_sourceImageRect.height() << "\"";
    // FIXME: should this dump also object returned by FEImage::image() ?

    ts << "]\n";
    return ts;
}

} // namespace WebCore
