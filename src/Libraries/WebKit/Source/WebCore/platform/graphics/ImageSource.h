/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include "ImageFrame.h"
#include "ImageOrientation.h"
#include "ImageTypes.h"
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class FragmentedSharedBuffer;

class ImageSource : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ImageSource> {
public:
    virtual ~ImageSource() = default;

    // Encoded and decoded data
    virtual EncodedDataStatus dataChanged(FragmentedSharedBuffer*, bool) { RELEASE_ASSERT_NOT_REACHED(); return EncodedDataStatus::Unknown; }
    virtual void destroyDecodedData(bool) { RELEASE_ASSERT_NOT_REACHED(); }

    // Animation
    virtual void startAnimation() { }
    virtual void stopAnimation() { }
    virtual void resetAnimation() { }
    virtual bool isAnimated() const { return false; }
    virtual bool isAnimating() const { return false; }
    virtual bool hasEverAnimated() const { return false; }

    // Decoding
    virtual bool isLargeForDecoding() const { return false; }
    virtual void stopDecodingWorkQueue() { RELEASE_ASSERT_NOT_REACHED(); }
    virtual void decode(Function<void(DecodingStatus)>&&)  { RELEASE_ASSERT_NOT_REACHED(); }

    // ImageFrame
    virtual unsigned currentFrameIndex() const { return primaryFrameIndex(); }

    virtual const ImageFrame& primaryImageFrame(const std::optional<SubsamplingLevel>& = std::nullopt) = 0;
    virtual const ImageFrame& currentImageFrame(const std::optional<SubsamplingLevel>& subsamplingLevel = std::nullopt) { return primaryImageFrame(subsamplingLevel); }

    // NativeImage
    virtual RefPtr<NativeImage> primaryNativeImage() = 0;
    virtual RefPtr<NativeImage> currentNativeImage() { return primaryNativeImage(); }
    virtual RefPtr<NativeImage> currentPreTransformedNativeImage(ImageOrientation) { return currentNativeImage(); }

    virtual RefPtr<NativeImage> nativeImageAtIndex(unsigned) { return primaryNativeImage(); }

    virtual Expected<Ref<NativeImage>, DecodingStatus> primaryNativeImageForDrawing(SubsamplingLevel, const DecodingOptions&);
    virtual Expected<Ref<NativeImage>, DecodingStatus> currentNativeImageForDrawing(SubsamplingLevel, const DecodingOptions&);

    // Image Metadata
    virtual IntSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const = 0;
    virtual IntSize sourceSize(ImageOrientation orientation = ImageOrientation::Orientation::FromImage) const { return size(orientation); }
    virtual bool hasDensityCorrectedSize() const { return false; }
    virtual ImageOrientation orientation() const { return ImageOrientation::Orientation::None; }
    virtual unsigned primaryFrameIndex() const { return 0; }
    virtual unsigned frameCount() const { return 1; }
    virtual DestinationColorSpace colorSpace() const = 0;
    virtual std::optional<Color> singlePixelSolidColor() const = 0;

    bool hasSolidColor() const;

    virtual String uti() const { return String(); }
    virtual String filenameExtension() const { return String(); }
    virtual String accessibilityDescription() const { return String(); }
    virtual std::optional<IntPoint> hotSpot() const { return { }; }

    virtual SubsamplingLevel subsamplingLevelForScaleFactor(GraphicsContext&, const FloatSize&, AllowImageSubsampling) { return SubsamplingLevel::Default; }

#if ENABLE(QUICKLOOK_FULLSCREEN)
    virtual bool shouldUseQuickLookForFullscreen() const { return false; }
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
    virtual bool isSpatial() const { return false; }
#endif

    // ImageFrame Metadata
    virtual Seconds frameDurationAtIndex(unsigned) const { RELEASE_ASSERT_NOT_REACHED(); return 0_s; }
    virtual ImageOrientation frameOrientationAtIndex(unsigned) const { RELEASE_ASSERT_NOT_REACHED(); return ImageOrientation::Orientation::None; }
    virtual Headroom frameHeadroomAtIndex(unsigned) const { RELEASE_ASSERT_NOT_REACHED(); return Headroom::None; }
    virtual DecodingStatus frameDecodingStatusAtIndex(unsigned) const { RELEASE_ASSERT_NOT_REACHED(); return DecodingStatus::Invalid; }

    // Testing support
    virtual unsigned decodeCountForTesting() const { return 0; }
    virtual unsigned blankDrawCountForTesting() const { return 0; }
    virtual void setMinimumDecodingDurationForTesting(Seconds) { RELEASE_ASSERT_NOT_REACHED(); }
    virtual void setClearDecoderAfterAsyncFrameRequestForTesting(bool) { RELEASE_ASSERT_NOT_REACHED(); }
    virtual void setAsyncDecodingEnabledForTesting(bool) { RELEASE_ASSERT_NOT_REACHED(); }
    virtual bool isAsyncDecodingEnabledForTesting() const { return false; }

    virtual void dump(WTF::TextStream&) const { }
};

} // namespace WebCore
