/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#include "ImageFrame.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {

ImageFrame::ImageFrame()
{
}

ImageFrame::ImageFrame(Ref<NativeImage>&& nativeImage)
    : m_nativeImage(WTFMove(nativeImage))
{
    m_size = m_nativeImage->size();
    m_hasAlpha = m_nativeImage->hasAlpha();
}

ImageFrame::~ImageFrame()
{
    clearImage();
}

const ImageFrame& ImageFrame::defaultFrame()
{
    static NeverDestroyed<ImageFrame> sharedInstance;
    return sharedInstance;
}

ImageFrame& ImageFrame::operator=(const ImageFrame& other)
{
    if (this == &other)
        return *this;

    m_decodingStatus = other.m_decodingStatus;
    m_size = other.m_size;

    m_nativeImage = other.m_nativeImage;
    m_subsamplingLevel = other.m_subsamplingLevel;
    m_decodingOptions = other.m_decodingOptions;

    m_orientation = other.m_orientation;
    m_duration = other.m_duration;
    m_hasAlpha = other.m_hasAlpha;
    return *this;
}

void ImageFrame::setDecodingStatus(DecodingStatus decodingStatus)
{
    ASSERT(decodingStatus != DecodingStatus::Decoding);
    m_decodingStatus = decodingStatus;
}

DecodingStatus ImageFrame::decodingStatus() const
{
    ASSERT(m_decodingStatus != DecodingStatus::Decoding);
    return m_decodingStatus;
}

unsigned ImageFrame::clearImage()
{
    if (!hasNativeImage())
        return 0;

    unsigned frameBytes = this->frameBytes();

    m_nativeImage->clearSubimages();
    m_nativeImage = nullptr;
    m_decodingOptions = DecodingOptions();

    return frameBytes;
}

unsigned ImageFrame::clear()
{
    unsigned frameBytes = clearImage();
    *this = ImageFrame();
    return frameBytes;
}

bool ImageFrame::hasNativeImage(const std::optional<SubsamplingLevel>& subsamplingLevel) const
{
    return m_nativeImage && (!subsamplingLevel || *subsamplingLevel >= m_subsamplingLevel);
}

bool ImageFrame::hasFullSizeNativeImage(const std::optional<SubsamplingLevel>& subsamplingLevel) const
{
    return hasNativeImage(subsamplingLevel) && m_decodingOptions.hasFullSize();
}

bool ImageFrame::hasDecodedNativeImageCompatibleWithOptions(const std::optional<SubsamplingLevel>& subsamplingLevel, const DecodingOptions& decodingOptions) const
{
    return isComplete() && hasNativeImage(subsamplingLevel) && m_decodingOptions.isCompatibleWith(decodingOptions);
}

} // namespace WebCore
