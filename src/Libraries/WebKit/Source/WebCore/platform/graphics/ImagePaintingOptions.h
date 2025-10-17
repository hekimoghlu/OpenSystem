/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include "DecodingOptions.h"
#include "GraphicsTypes.h"
#include "ImageOrientation.h"
#include "ImageTypes.h"
#include <initializer_list>
#include <wtf/Forward.h>

namespace WebCore {

struct ImagePaintingOptions {
    template<typename Type> static constexpr bool isOptionType =
        std::is_same_v<Type, CompositeOperator>
        || std::is_same_v<Type, BlendMode>
        || std::is_same_v<Type, DecodingMode>
        || std::is_same_v<Type, ImageOrientation>
        || std::is_same_v<Type, ImageOrientation::Orientation>
        || std::is_same_v<Type, InterpolationQuality>
        || std::is_same_v<Type, AllowImageSubsampling>
#if USE(SKIA)
        || std::is_same_v<Type, StrictImageClamping>
#endif
        || std::is_same_v<Type, ShowDebugBackground>
        || std::is_same_v<Type, Headroom>;

    // This is a single-argument initializer to support pattern of
    // ImageDrawResult drawImage(..., ImagePaintingOptions = { ImageOrientation::Orientation::FromImage });
    // Should be removed once the pattern is not so prevalent.
    template<typename T, typename = std::enable_if_t<isOptionType<std::decay_t<T>>>>
    ImagePaintingOptions(std::initializer_list<T> options)
    {
        for (auto& option : options)
            setOption(option);
    }
    template<typename T, typename = std::enable_if_t<isOptionType<std::decay_t<T>>>>
    explicit ImagePaintingOptions(T option)
    {
        setOption(option);
    }

    template<typename T, typename U, typename... Rest, typename = std::enable_if_t<isOptionType<std::decay_t<T>>>>
    ImagePaintingOptions(T first, U second, Rest... rest)
    {
        setOption(first);
        setOption(second);
        (setOption(rest), ...);
    }

    template<typename... Overrides>
    ImagePaintingOptions(const ImagePaintingOptions& other, Overrides... overrides)
        : ImagePaintingOptions(other)
    {
        (setOption(overrides), ...);
    }

    ImagePaintingOptions() = default;
    ImagePaintingOptions(const ImagePaintingOptions&) = default;
    ImagePaintingOptions(ImagePaintingOptions&&) = default;
    ImagePaintingOptions& operator=(const ImagePaintingOptions&) = default;
    ImagePaintingOptions& operator=(ImagePaintingOptions&&) = default;

    CompositeOperator compositeOperator() const { return m_compositeOperator; }
    BlendMode blendMode() const { return m_blendMode; }
    DecodingMode decodingMode() const { return m_decodingMode; }
    ImageOrientation orientation() const { return m_orientation; }
    InterpolationQuality interpolationQuality() const { return m_interpolationQuality; }
    AllowImageSubsampling allowImageSubsampling() const { return m_allowImageSubsampling; }
#if USE(SKIA)
    StrictImageClamping strictImageClamping() const { return m_strictImageClamping; }
#endif
    ShowDebugBackground showDebugBackground() const { return m_showDebugBackground; }
    Headroom headroom() const { return m_headroom; }

private:
    void setOption(CompositeOperator compositeOperator) { m_compositeOperator = compositeOperator; }
    void setOption(BlendMode blendMode) { m_blendMode = blendMode; }
    void setOption(DecodingMode decodingMode) { m_decodingMode = decodingMode; }
    void setOption(ImageOrientation orientation) { m_orientation = orientation.orientation(); }
    void setOption(ImageOrientation::Orientation orientation) { m_orientation = orientation; }
    void setOption(InterpolationQuality interpolationQuality) { m_interpolationQuality = interpolationQuality; }
    void setOption(AllowImageSubsampling allowImageSubsampling) { m_allowImageSubsampling = allowImageSubsampling; }
#if USE(SKIA)
    void setOption(StrictImageClamping strictImageClamping) { m_strictImageClamping = strictImageClamping; }
#endif
    void setOption(ShowDebugBackground showDebugBackground) { m_showDebugBackground = showDebugBackground; }
    void setOption(Headroom headroom) { m_headroom = headroom; }

    BlendMode m_blendMode : 5 { BlendMode::Normal };
    DecodingMode m_decodingMode : 3 { DecodingMode::Synchronous };
    CompositeOperator m_compositeOperator : 4 { CompositeOperator::SourceOver };
    ImageOrientation::Orientation m_orientation : 4 { ImageOrientation::Orientation::None };
    InterpolationQuality m_interpolationQuality : 4 { InterpolationQuality::Default };
    AllowImageSubsampling m_allowImageSubsampling : 1 { AllowImageSubsampling::No };
#if USE(SKIA)
    StrictImageClamping m_strictImageClamping: 1 { StrictImageClamping::Yes };
#endif
    ShowDebugBackground m_showDebugBackground : 1 { ShowDebugBackground::No };
    Headroom m_headroom { Headroom::FromImage };
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, ImagePaintingOptions);

} // namespace WebCore
