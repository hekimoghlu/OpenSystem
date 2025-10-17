/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#pragma once

#include "FilterEffect.h"
#include "Image.h"
#include "ImageBuffer.h"
#include "SVGPreserveAspectRatioValue.h"
#include "SourceImage.h"

namespace WebCore {

class Image;
class ImageBuffer;

class FEImage final : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEImage> create(SourceImage&&, const FloatRect& sourceImageRect, const SVGPreserveAspectRatioValue&);

    bool operator==(const FEImage&) const;

    const SourceImage& sourceImage() const { return m_sourceImage; }
    void setImageSource(SourceImage&& sourceImage) { m_sourceImage = WTFMove(sourceImage); }

    FloatRect sourceImageRect() const { return m_sourceImageRect; }
    const SVGPreserveAspectRatioValue& preserveAspectRatio() const { return m_preserveAspectRatio; }

private:
    FEImage(SourceImage&&, const FloatRect& sourceImageRect, const SVGPreserveAspectRatioValue&);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEImage>(*this, other); }

    unsigned numberOfEffectInputs() const override { return 0; }

    // FEImage results are always in DestinationColorSpace::SRGB()
    void setOperatingColorSpace(const DestinationColorSpace&) override { }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const final;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const final;

    SourceImage m_sourceImage;
    FloatRect m_sourceImageRect;
    SVGPreserveAspectRatioValue m_preserveAspectRatio;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEImage)
