/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

#include "GeneratedImage.h"

namespace WebCore {

class Gradient;
class ImageBuffer;

class GradientImage final : public GeneratedImage {
public:
    static Ref<GradientImage> create(Gradient& generator, const FloatSize& size)
    {
        return adoptRef(*new GradientImage(generator, size));
    }

    virtual ~GradientImage();

    const Gradient& gradient() const { return m_gradient.get(); }

private:
    WEBCORE_EXPORT GradientImage(Gradient&, const FloatSize&);

    ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) final;
    void drawPattern(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;
    bool isGradientImage() const final { return true; }
    void dump(WTF::TextStream&) const final;
    
    Ref<Gradient> m_gradient;
    RefPtr<ImageBuffer> m_cachedImage;
    FloatSize m_cachedAdjustedSize;
    unsigned m_cachedGeneratorHash { 0 };
    FloatSize m_cachedScaleFactor;
};

}

SPECIALIZE_TYPE_TRAITS_IMAGE(GradientImage)
