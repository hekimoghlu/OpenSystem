/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include "Color.h"
#include "DestinationColorSpace.h"
#include "ImageOrientation.h"
#include "ImageTypes.h"
#include "IntPoint.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class BitmapImageSource;
class GraphicsContext;
class ImageDecoder;
class ImageFrame;
class NativeImage;

class BitmapImageDescriptor {
public:
    BitmapImageDescriptor(BitmapImageSource&);

    void clear() { m_cachedFlags = { }; }

    EncodedDataStatus encodedDataStatus() const;
    IntSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const;
    IntSize sourceSize(ImageOrientation = ImageOrientation::Orientation::FromImage) const;
    std::optional<IntSize> densityCorrectedSize() const;
    ImageOrientation orientation() const;
    unsigned primaryFrameIndex() const;
    unsigned frameCount() const;
    RepetitionCount repetitionCount() const;
    DestinationColorSpace colorSpace() const;
    std::optional<Color> singlePixelSolidColor() const;

    String uti() const;
    String filenameExtension() const;
    String accessibilityDescription() const;
    std::optional<IntPoint> hotSpot() const;
    SubsamplingLevel maximumSubsamplingLevel() const;
    SubsamplingLevel subsamplingLevelForScaleFactor(GraphicsContext&, const FloatSize& scaleFactor, AllowImageSubsampling) const;

#if ENABLE(QUICKLOOK_FULLSCREEN)
    bool shouldUseQuickLookForFullscreen() const;
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
    bool isSpatial() const;
#endif

    void dump(TextStream&) const;

private:
    enum class CachedFlag : uint16_t {
        EncodedDataStatus           = 1 << 0,
        Size                        = 1 << 1,
        DensityCorrectedSize        = 1 << 2,
        Orientation                 = 1 << 3,
        PrimaryFrameIndex           = 1 << 4,
        FrameCount                  = 1 << 5,
        RepetitionCount             = 1 << 6,
        ColorSpace                  = 1 << 7,
        SinglePixelSolidColor       = 1 << 8,

        UTI                         = 1 << 9,
        FilenameExtension           = 1 << 10,
        AccessibilityDescription    = 1 << 11,
        HotSpot                     = 1 << 12,
        MaximumSubsamplingLevel     = 1 << 13,
    };

    template<typename MetadataType>
    MetadataType imageMetadata(MetadataType& cachedValue, const MetadataType& defaultValue, CachedFlag, MetadataType (ImageDecoder::*functor)() const) const;

    template<typename MetadataType>
    MetadataType primaryImageFrameMetadata(MetadataType& cachedValue, CachedFlag, MetadataType (ImageFrame::*functor)() const, const std::optional<SubsamplingLevel>& = std::nullopt) const;

    template<typename MetadataType>
    MetadataType primaryNativeImageMetadata(MetadataType& cachedValue, const MetadataType& defaultValue, CachedFlag, MetadataType (NativeImage::*functor)() const) const;

    mutable OptionSet<CachedFlag> m_cachedFlags;

    mutable EncodedDataStatus m_encodedDataStatus { EncodedDataStatus::Unknown };
    mutable IntSize m_size;
    mutable std::optional<IntSize> m_densityCorrectedSize;
    mutable ImageOrientation m_orientation;
    mutable size_t m_primaryFrameIndex { 0 };
    mutable size_t m_frameCount { 0 };
    mutable RepetitionCount m_repetitionCount { RepetitionCountNone };
    mutable DestinationColorSpace m_colorSpace { DestinationColorSpace::SRGB() };
    mutable std::optional<Color> m_singlePixelSolidColor;

    mutable String m_uti;
    mutable String m_filenameExtension;
    mutable String m_accessibilityDescription;
    mutable std::optional<IntPoint> m_hotSpot;
    mutable SubsamplingLevel m_maximumSubsamplingLevel { SubsamplingLevel::Default };

    BitmapImageSource& m_source;
};

} // namespace WebCore
