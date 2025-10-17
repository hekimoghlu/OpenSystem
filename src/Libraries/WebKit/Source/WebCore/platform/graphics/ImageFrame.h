/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#include "DecodingOptions.h"
#include "ImageOrientation.h"
#include "ImageTypes.h"
#include "IntSize.h"
#include "NativeImage.h"
#include <wtf/Seconds.h>

// X11 headers define a bunch of macros with common terms, interfering with WebCore and WTF enum values.
// As a workaround, we explicitly undef them here.
#if defined(None)
#undef None
#endif

namespace WebCore {

class ImageFrame {
    friend class BitmapImageSource;
    friend class ImageDecoder;
    friend class ImageDecoderCG;
public:
    enum class Caching { Metadata, MetadataAndImage };

    ImageFrame();
    ImageFrame(Ref<NativeImage>&&);
    ImageFrame(const ImageFrame& other) { operator=(other); }

    ~ImageFrame();

    static const ImageFrame& defaultFrame();

    ImageFrame& operator=(const ImageFrame& other);

    unsigned clearImage();
    unsigned clear();

    void setDecodingStatus(DecodingStatus);
    DecodingStatus decodingStatus() const;

    bool isInvalid() const { return m_decodingStatus == DecodingStatus::Invalid; }
    bool isPartial() const { return m_decodingStatus == DecodingStatus::Partial; }
    bool isComplete() const { return m_decodingStatus == DecodingStatus::Complete; }

    void setSize(const IntSize& size) { m_size = size; }
    IntSize size() const { return m_size; }

    unsigned frameBytes() const { return hasNativeImage() ? (size().area() * sizeof(uint32_t)).value() : 0; }
    SubsamplingLevel subsamplingLevel() const { return m_subsamplingLevel; }
    DecodingOptions decodingOptions() const { return m_decodingOptions; }

    RefPtr<NativeImage> nativeImage() const { return m_nativeImage; }

    void setOrientation(ImageOrientation orientation) { m_orientation = orientation; };
    ImageOrientation orientation() const { return m_orientation; }

    void setHeadroom(Headroom headroom) { m_headroom = headroom; };
    Headroom headroom() const { return m_headroom; }

    void setDensityCorrectedSize(const IntSize& size) { m_densityCorrectedSize = size; }
    std::optional<IntSize> densityCorrectedSize() const { return m_densityCorrectedSize; }

    void setDuration(const Seconds& duration) { m_duration = duration; }
    Seconds duration() const { return m_duration; }

    void setHasAlpha(bool hasAlpha) { m_hasAlpha = hasAlpha; }
    bool hasAlpha() const { return !hasMetadata() || m_hasAlpha; }

    bool hasNativeImage(const std::optional<SubsamplingLevel>& = { }) const;
    bool hasFullSizeNativeImage(const std::optional<SubsamplingLevel>& = { }) const;
    bool hasDecodedNativeImageCompatibleWithOptions(const std::optional<SubsamplingLevel>&, const DecodingOptions&) const;
    bool hasMetadata() const { return !size().isEmpty(); }

private:
    DecodingStatus m_decodingStatus { DecodingStatus::Invalid };
    IntSize m_size;

    RefPtr<NativeImage> m_nativeImage;
    SubsamplingLevel m_subsamplingLevel { SubsamplingLevel::Default };
    DecodingOptions m_decodingOptions { DecodingMode::Auto };

    ImageOrientation m_orientation { ImageOrientation::Orientation::None };
    Headroom m_headroom { Headroom::None };
    std::optional<IntSize> m_densityCorrectedSize;
    Seconds m_duration;
    bool m_hasAlpha { true };
};

} // namespace WebCore
