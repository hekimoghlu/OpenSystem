/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
#include "ImageSource.h"
#include <wtf/Function.h>

namespace WebCore {

class ImageObserver;
class ImageSource;
class NativeImage;

class BitmapImage final : public Image {
public:
    WEBCORE_EXPORT static Ref<BitmapImage> create(ImageObserver* = nullptr, AlphaOption = AlphaOption::Premultiplied, GammaAndColorProfileOption = GammaAndColorProfileOption::Applied);
    WEBCORE_EXPORT static Ref<BitmapImage> create(Ref<NativeImage>&&);
    WEBCORE_EXPORT static RefPtr<BitmapImage> create(RefPtr<NativeImage>&&);
    WEBCORE_EXPORT static RefPtr<BitmapImage> create(PlatformImagePtr&&);

    // Animation
    void startAnimation() final { m_source->startAnimation(); }
    void stopAnimation() final { m_source->stopAnimation(); }
    void resetAnimation() final { m_source->resetAnimation(); }
    bool isAnimated() const final { return m_source->isAnimated(); }
    bool hasEverAnimated() const { return m_source->hasEverAnimated(); }

    // Decoding
    bool isLargeForDecoding() const { return m_source->isLargeForDecoding(); }
    void stopDecodingWorkQueue() { m_source->stopDecodingWorkQueue(); }
    void decode(Function<void(DecodingStatus)>&& decodeCallback) { m_source->decode(WTFMove(decodeCallback)); }

    // Current ImageFrame
    unsigned currentFrameIndex() const { return m_source->currentFrameIndex(); }
    bool currentFrameHasAlpha() const { return m_source->currentImageFrame().hasAlpha(); }
    ImageOrientation currentFrameOrientation() const { return m_source->currentImageFrame().orientation(); }
    Headroom currentFrameHeadroom() const { return m_source->currentImageFrame().headroom(); }
    DecodingOptions currentFrameDecodingOptions() const { return m_source->currentImageFrame().decodingOptions(); }

    // Primary & current NativeImage
    RefPtr<NativeImage> primaryNativeImage() { return m_source->primaryNativeImage(); }
    RefPtr<NativeImage> nativeImage(const DestinationColorSpace& = DestinationColorSpace::SRGB()) final { return primaryNativeImage(); }
    RefPtr<NativeImage> currentNativeImage() final { return m_source->currentNativeImage(); }

    // Image Metadata
    FloatSize size(ImageOrientation orientation = ImageOrientation::Orientation::FromImage) const final { return m_source->size(orientation); }
    FloatSize sourceSize(ImageOrientation orientation = ImageOrientation::Orientation::FromImage) const { return m_source->sourceSize(orientation); }
    DestinationColorSpace colorSpace() final { return m_source->colorSpace(); }
    ImageOrientation orientation() const final { return m_source->orientation(); }
    unsigned frameCount() const final { return m_source->frameCount(); }
#if ASSERT_ENABLED
    bool hasSolidColor() final { return m_source->hasSolidColor(); }
#endif

    // ImageFrame
    Seconds frameDurationAtIndex(unsigned index) const { return m_source->frameDurationAtIndex(index); }

    // NativeImage
    RefPtr<NativeImage> nativeImageAtIndex(unsigned index) final { return m_source->nativeImageAtIndex(index); }

    // Testing support.
    CString sourceUTF8() const { return sourceURL().string().utf8(); }
    void setAsyncDecodingEnabledForTesting(bool enabled) { m_source->setAsyncDecodingEnabledForTesting(enabled); }
    bool isAsyncDecodingEnabledForTesting() const { return m_source->isAsyncDecodingEnabledForTesting(); }
    void setMinimumDecodingDurationForTesting(Seconds duration) { m_source->setMinimumDecodingDurationForTesting(duration); }
    void setClearDecoderAfterAsyncFrameRequestForTesting(bool enabled) { m_source->setClearDecoderAfterAsyncFrameRequestForTesting(enabled); }
    unsigned decodeCountForTesting() const { return m_source->decodeCountForTesting(); }
    unsigned blankDrawCountForTesting() const { return m_source->blankDrawCountForTesting(); }

private:
    BitmapImage(ImageObserver*, AlphaOption, GammaAndColorProfileOption);
    BitmapImage(Ref<NativeImage>&&);

    // Encoded and decoded data
    EncodedDataStatus dataChanged(bool allDataReceived) final;
    void destroyDecodedData(bool destroyAll = true) final;

    // Current ImageFrame
    bool currentFrameKnownToBeOpaque() const final { return !currentFrameHasAlpha(); }

    // Current NativeImage
    RefPtr<NativeImage> currentPreTransformedNativeImage(ImageOrientation orientation) final { return m_source->currentPreTransformedNativeImage(orientation); }

    // Image Metadata
    bool hasDensityCorrectedSize() const final { return m_source->hasDensityCorrectedSize(); }
    String uti() const final { return m_source->uti(); }
    String filenameExtension() const final { return m_source->filenameExtension(); }
    String accessibilityDescription() const final { return m_source->accessibilityDescription(); }
    std::optional<IntPoint> hotSpot() const final { return m_source->hotSpot(); }
    std::optional<Color> singlePixelSolidColor() const final { return m_source->singlePixelSolidColor(); }

#if ENABLE(QUICKLOOK_FULLSCREEN)
    bool shouldUseQuickLookForFullscreen() const final { return m_source->shouldUseQuickLookForFullscreen(); }
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
    bool isSpatial() const final { return m_source->isSpatial(); }
#endif

    // Image methods
    bool isBitmapImage() const final { return true; }
    bool isAnimating() const final { return m_source->isAnimating(); }

    ImageDrawResult draw(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions = { }) final;
    void drawPattern(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& tileRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;
    void drawLuminanceMaskPattern(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& tileRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions);

    void dump(WTF::TextStream&) const final;

    Ref<ImageSource> m_source;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_IMAGE(BitmapImage)
