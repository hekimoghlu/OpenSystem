/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#include "AffineTransform.h"
#include "FloatRect.h"
#include "FloatSize.h"
#include "Image.h"
#include "SVGImage.h"
#include <wtf/URL.h>

namespace WebCore {

class SVGImageForContainer final : public Image {
public:
    static Ref<SVGImageForContainer> create(SVGImage* image, const FloatSize& containerSize, float containerZoom, const URL& initialFragmentURL)
    {
        return adoptRef(*new SVGImageForContainer(image, containerSize, containerZoom, initialFragmentURL));
    }

    bool isSVGImageForContainer() const final { return true; }

    FloatSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const final;

    bool usesContainerSize() const final { return m_image->usesContainerSize(); }
    bool hasRelativeWidth() const final { return m_image->hasRelativeWidth(); }
    bool hasRelativeHeight() const final { return m_image->hasRelativeHeight(); }
    void computeIntrinsicDimensions(Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) final
    {
        protectedImage()->computeIntrinsicDimensions(intrinsicWidth, intrinsicHeight, intrinsicRatio);
    }

    ImageDrawResult draw(GraphicsContext&, const FloatRect&, const FloatRect&, ImagePaintingOptions = { }) final;

    void drawPattern(GraphicsContext&, const FloatRect&, const FloatRect&, const AffineTransform&, const FloatPoint&, const FloatSize&, ImagePaintingOptions = { }) final;

    // FIXME: Implement this to be less conservative.
    bool currentFrameKnownToBeOpaque() const final { return false; }

    RefPtr<NativeImage> currentNativeImage() final;

private:
    WEBCORE_EXPORT SVGImageForContainer(SVGImage*, const FloatSize& containerSize, float containerZoom, const URL& initialFragmentURL);
    RefPtr<SVGImage> protectedImage() const;

    WeakPtr<SVGImage> m_image;
    const FloatSize m_containerSize;
    const float m_containerZoom;
    const URL m_initialFragmentURL;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_IMAGE(SVGImageForContainer)
