/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

#include "FloatSize.h"
#include "GeneratedImage.h"
#include "Image.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class CrossfadeGeneratedImage final : public GeneratedImage {
public:
    static Ref<CrossfadeGeneratedImage> create(Image& fromImage, Image& toImage, float percentage, const FloatSize& crossfadeSize, const FloatSize& size)
    {
        return adoptRef(*new CrossfadeGeneratedImage(fromImage, toImage, percentage, crossfadeSize, size));
    }

    void setContainerSize(const FloatSize&) override { }
    bool usesContainerSize() const override { return false; }
    bool hasRelativeWidth() const override { return false; }
    bool hasRelativeHeight() const override { return false; }

    FloatSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const override { return m_crossfadeSize; }

private:
    ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) override;
    void drawPattern(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) override;

    CrossfadeGeneratedImage(Image& fromImage, Image& toImage, float percentage, const FloatSize& crossfadeSize, const FloatSize&);

    bool isCrossfadeGeneratedImage() const override { return true; }
    void dump(WTF::TextStream&) const override;
    
    void drawCrossfade(GraphicsContext&);

    Ref<Image> m_fromImage;
    Ref<Image> m_toImage;

    float m_percentage;
    FloatSize m_crossfadeSize;
};

}

SPECIALIZE_TYPE_TRAITS_IMAGE(CrossfadeGeneratedImage)
