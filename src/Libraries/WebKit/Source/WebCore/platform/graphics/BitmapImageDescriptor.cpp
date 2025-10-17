/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#include "BitmapImageDescriptor.h"

#include "BitmapImageSource.h"
#include "GraphicsContext.h"
#include "ImageDecoder.h"

namespace WebCore {

BitmapImageDescriptor::BitmapImageDescriptor(BitmapImageSource& source)
    : m_source(source)
{
}

template<typename MetadataType>
MetadataType BitmapImageDescriptor::imageMetadata(MetadataType& cachedValue, const MetadataType& defaultValue, CachedFlag cachedFlag, MetadataType (ImageDecoder::*functor)() const) const
{
    if (m_cachedFlags.contains(cachedFlag))
        return cachedValue;

    auto decoder = m_source.decoderIfExists();
    if (!decoder)
        return defaultValue;

    if (!decoder->isSizeAvailable())
        return defaultValue;

    cachedValue = (*decoder.*functor)();
    m_cachedFlags.add(cachedFlag);
    const_cast<BitmapImageSource&>(m_source).didDecodeProperties(decoder->bytesDecodedToDetermineProperties());
    return cachedValue;
}

template<typename MetadataType>
MetadataType BitmapImageDescriptor::primaryImageFrameMetadata(MetadataType& cachedValue, CachedFlag cachedFlag, MetadataType (ImageFrame::*functor)() const, const std::optional<SubsamplingLevel>& subsamplingLevel) const
{
    if (m_cachedFlags.contains(cachedFlag))
        return cachedValue;

    auto& frame = const_cast<BitmapImageSource&>(m_source).primaryImageFrame(subsamplingLevel);

    // Don't cache any unavailable frame metadata. Just return the default metadata.
    if (!frame.hasMetadata())
        return (frame.*functor)();

    cachedValue = (frame.*functor)();
    m_cachedFlags.add(cachedFlag);
    return cachedValue;
}

template<typename MetadataType>
MetadataType BitmapImageDescriptor::primaryNativeImageMetadata(MetadataType& cachedValue, const MetadataType& defaultValue, CachedFlag cachedFlag, MetadataType (NativeImage::*functor)() const) const
{
    if (m_cachedFlags.contains(cachedFlag))
        return cachedValue;

    RefPtr nativeImage = const_cast<BitmapImageSource&>(m_source).primaryNativeImage();
    if (!nativeImage)
        return defaultValue;

    cachedValue = (*nativeImage.*functor)();
    m_cachedFlags.add(cachedFlag);
    return cachedValue;
}

EncodedDataStatus BitmapImageDescriptor::encodedDataStatus() const
{
    return imageMetadata(m_encodedDataStatus, EncodedDataStatus::Unknown, CachedFlag::EncodedDataStatus, &ImageDecoder::encodedDataStatus);
}

IntSize BitmapImageDescriptor::size(ImageOrientation orientation) const
{
    auto densityCorrectedSize = this->densityCorrectedSize();
    if (!densityCorrectedSize)
        return sourceSize(orientation);

    if (orientation == ImageOrientation::Orientation::FromImage)
        orientation = this->orientation();

    return orientation.usesWidthAsHeight() ? densityCorrectedSize->transposedSize() : *densityCorrectedSize;
}

IntSize BitmapImageDescriptor::sourceSize(ImageOrientation orientation) const
{
    IntSize size;

#if !USE(CG)
    // It's possible that we have decoded the metadata, but not frame contents yet. In that case ImageDecoder claims to
    // have the size available, but the frame cache is empty. Return the decoder size without caching in such case.
    auto decoder = m_source.decoderIfExists();
    if (decoder && m_source.frames().isEmpty())
        size = decoder->size();
    else
#endif
        size = primaryImageFrameMetadata(m_size, CachedFlag::Size, &ImageFrame::size, SubsamplingLevel::Default);

    if (orientation == ImageOrientation::Orientation::FromImage)
        orientation = this->orientation();

    return orientation.usesWidthAsHeight() ? size.transposedSize() : size;
}

std::optional<IntSize> BitmapImageDescriptor::densityCorrectedSize() const
{
    return primaryImageFrameMetadata(m_densityCorrectedSize, CachedFlag::DensityCorrectedSize, &ImageFrame::densityCorrectedSize, SubsamplingLevel::Default);
}

ImageOrientation BitmapImageDescriptor::orientation() const
{
    return primaryImageFrameMetadata(m_orientation, CachedFlag::Orientation, &ImageFrame::orientation);
}

unsigned BitmapImageDescriptor::primaryFrameIndex() const
{
    return imageMetadata(m_primaryFrameIndex, std::size_t(0), CachedFlag::PrimaryFrameIndex, &ImageDecoder::primaryFrameIndex);
}

unsigned BitmapImageDescriptor::frameCount() const
{
    return imageMetadata(m_frameCount, std::size_t(0), CachedFlag::FrameCount, &ImageDecoder::frameCount);
}

RepetitionCount BitmapImageDescriptor::repetitionCount() const
{
    return imageMetadata(m_repetitionCount, static_cast<RepetitionCount>(RepetitionCountNone), CachedFlag::RepetitionCount, &ImageDecoder::repetitionCount);
}

DestinationColorSpace BitmapImageDescriptor::colorSpace() const
{
    return primaryNativeImageMetadata(m_colorSpace, DestinationColorSpace::SRGB(), CachedFlag::ColorSpace, &NativeImage::colorSpace);
}

std::optional<Color> BitmapImageDescriptor::singlePixelSolidColor() const
{
    if (!m_source.hasSolidColor())
        return std::nullopt;

    return primaryNativeImageMetadata(m_singlePixelSolidColor, std::optional<Color>(), CachedFlag::SinglePixelSolidColor, &NativeImage::singlePixelSolidColor);
}

String BitmapImageDescriptor::uti() const
{
#if USE(CG)
    return imageMetadata(m_uti, String(), CachedFlag::UTI, &ImageDecoder::uti);
#else
    return String();
#endif
}

String BitmapImageDescriptor::filenameExtension() const
{
    return imageMetadata(m_filenameExtension, String(), CachedFlag::FilenameExtension, &ImageDecoder::filenameExtension);
}

String BitmapImageDescriptor::accessibilityDescription() const
{
    return imageMetadata(m_accessibilityDescription, String(), CachedFlag::AccessibilityDescription, &ImageDecoder::accessibilityDescription);
}

std::optional<IntPoint> BitmapImageDescriptor::hotSpot() const
{
    return imageMetadata(m_hotSpot, std::optional<IntPoint>(), CachedFlag::HotSpot, &ImageDecoder::hotSpot);
}

SubsamplingLevel BitmapImageDescriptor::maximumSubsamplingLevel() const
{
    if (m_cachedFlags.contains(CachedFlag::MaximumSubsamplingLevel))
        return m_maximumSubsamplingLevel;

    auto decoder = m_source.decoderIfExists();
    if (!decoder)
        return SubsamplingLevel::Default;

    if (!decoder->isSizeAvailable())
        return SubsamplingLevel::Default;

    // FIXME: this value was chosen to be appropriate for Apple ports since the image
    // subsampling is only enabled by default on Apple ports. Choose a different value
    // if image subsampling is enabled on other platform.
    static constexpr int maximumImageAreaBeforeSubsampling = 5 * 1024 * 1024;
    auto level = SubsamplingLevel::First;

    for (; level < SubsamplingLevel::Last; ++level) {
        auto area = m_source.frameSizeAtIndex(0, level).unclampedArea();
        if (area < maximumImageAreaBeforeSubsampling)
            break;
    }

    m_maximumSubsamplingLevel = level;
    m_cachedFlags.add(CachedFlag::MaximumSubsamplingLevel);
    return m_maximumSubsamplingLevel;
}

SubsamplingLevel BitmapImageDescriptor::subsamplingLevelForScaleFactor(GraphicsContext& context, const FloatSize& scaleFactor, AllowImageSubsampling allowImageSubsampling) const
{
#if USE(CG)
    if (allowImageSubsampling == AllowImageSubsampling::No)
        return SubsamplingLevel::Default;

    // Never use subsampled images for drawing into PDF contexts.
    if (context.hasPlatformContext() && CGContextGetType(context.platformContext()) == kCGContextTypePDF)
        return SubsamplingLevel::Default;

    float scale = std::min(float(1), std::max(scaleFactor.width(), scaleFactor.height()));
    if (!(scale > 0 && scale <= 1))
        return SubsamplingLevel::Default;

    int result = std::ceil(std::log2(1 / scale));
    return static_cast<SubsamplingLevel>(std::min(result, static_cast<int>(maximumSubsamplingLevel())));
#else
    UNUSED_PARAM(context);
    UNUSED_PARAM(scaleFactor);
    UNUSED_PARAM(allowImageSubsampling);
    return SubsamplingLevel::Default;
#endif
}

#if ENABLE(QUICKLOOK_FULLSCREEN)
bool BitmapImageDescriptor::shouldUseQuickLookForFullscreen() const
{
    if (auto decoder = m_source.decoderIfExists())
        return decoder->shouldUseQuickLookForFullscreen();
    return false;
}
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
bool BitmapImageDescriptor::isSpatial() const
{
    if (RefPtr decoder = m_source.decoderIfExists())
        return decoder->isSpatial();
    return false;
}
#endif

void BitmapImageDescriptor::dump(TextStream& ts) const
{
    ts.dumpProperty("size", size());
    ts.dumpProperty("density-corrected-size", densityCorrectedSize());
    ts.dumpProperty("primary-frame-index", primaryFrameIndex());
    ts.dumpProperty("frame-count", frameCount());
    ts.dumpProperty("repetition-count", repetitionCount());

    ts.dumpProperty("uti", uti());
    ts.dumpProperty("filename-extension", filenameExtension());
    ts.dumpProperty("accessibility-description", accessibilityDescription());
}

} // namespace WebCore
