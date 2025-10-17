/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#include "Image.h"
#include <wtf/URL.h>

namespace WebCore {

class Element;
class ImageBuffer;
class LocalFrameView;
class Page;
class RenderBox;
class SVGSVGElement;
class SVGImageChromeClient;
class SVGImageForContainer;
class Settings;

class SVGImage final : public Image {
public:
    static Ref<SVGImage> create(ImageObserver& observer) { return adoptRef(*new SVGImage(observer)); }
    WEBCORE_EXPORT static bool isDataDecodable(const Settings&, std::span<const uint8_t>);

    RenderBox* embeddedContentBox() const;
    LocalFrameView* frameView() const;
    RefPtr<LocalFrameView> protectedFrameView() const;

    bool isSVGImage() const final { return true; }
    FloatSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const final { return m_intrinsicSize; }

    bool renderingTaintsOrigin() const final;

    bool hasRelativeWidth() const final;
    bool hasRelativeHeight() const final;

    // Start the animation from the beginning.
    void startAnimation() final;
    // Resume the animation from where it was last stopped.
    void resumeAnimation();
    void stopAnimation() final;
    void resetAnimation() final;
    bool isAnimating() const final;

    void scheduleStartAnimation();

    Page* internalPage() { return m_page.get(); }
    WEBCORE_EXPORT RefPtr<SVGSVGElement> rootElement() const;

private:
    friend class SVGImageChromeClient;
    friend class SVGImageForContainer;

    virtual ~SVGImage();

    String filenameExtension() const final;

    void setContainerSize(const FloatSize&) final;
    IntSize containerSize() const;
    bool usesContainerSize() const final { return true; }
    void computeIntrinsicDimensions(Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) final;

    void reportApproximateMemoryCost() const;
    EncodedDataStatus dataChanged(bool allDataReceived) final;

    // FIXME: SVGImages will be unable to prune because destroyDecodedData() is not implemented yet.

    // FIXME: Implement this to be less conservative.
    bool currentFrameKnownToBeOpaque() const final { return false; }

    RefPtr<NativeImage> nativeImage(const DestinationColorSpace& = DestinationColorSpace::SRGB()) final;

    void startAnimationTimerFired();

    WEBCORE_EXPORT explicit SVGImage(ImageObserver&);
    ImageDrawResult draw(GraphicsContext&, const FloatRect& destination, const FloatRect& source, ImagePaintingOptions = { }) final;
    ImageDrawResult drawForContainer(GraphicsContext&, const FloatSize containerSize, float containerZoom, const URL& initialFragmentURL, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { });
    void drawPatternForContainer(GraphicsContext&, const FloatSize& containerSize, float containerZoom, const URL& initialFragmentURL, const FloatRect& srcRect, const AffineTransform&, const FloatPoint& phase, const FloatSize& spacing, const FloatRect&, ImagePaintingOptions = { });

    RefPtr<Page> m_page;
    FloatSize m_intrinsicSize;

    Timer m_startAnimationTimer;
};

bool isInSVGImage(const Element*);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_IMAGE(SVGImage)
