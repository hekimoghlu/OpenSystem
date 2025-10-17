/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#include "Image.h"

namespace WebCore {

class GeneratedImage : public Image {
public:
    void setContainerSize(const FloatSize& size) override { m_size = size; }
    bool usesContainerSize() const override { return true; }
    bool hasRelativeWidth() const override { return true; }
    bool hasRelativeHeight() const override { return true; }
    void computeIntrinsicDimensions(Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) override;

    FloatSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const override { return m_size; }

protected:
    ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) override = 0;
    void drawPattern(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) override = 0;

    // FIXME: Implement this to be less conservative.
    bool currentFrameKnownToBeOpaque() const override { return false; }

    GeneratedImage() = default;

private:
    bool isGeneratedImage() const override { return true; }

    FloatSize m_size;
};

}

SPECIALIZE_TYPE_TRAITS_IMAGE(GeneratedImage)
