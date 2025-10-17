/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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

#include "ImageSource.h"

namespace WebCore {

class NativeImageSource : public ImageSource {
public:
    static Ref<NativeImageSource> create(Ref<NativeImage>&&);

private:
    NativeImageSource(Ref<NativeImage>&&);

    IntSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const final { return m_frame.size(); }
    DestinationColorSpace colorSpace() const final { return m_frame.nativeImage()->colorSpace(); }
    std::optional<Color> singlePixelSolidColor() const final { return m_frame.nativeImage()->singlePixelSolidColor(); }

    const ImageFrame& primaryImageFrame(const std::optional<SubsamplingLevel>& = std::nullopt) final { return m_frame; }

    RefPtr<NativeImage> primaryNativeImage() final { return m_frame.nativeImage(); }

    void dump(TextStream&) const final;

    ImageFrame m_frame;
};

} // namespace WebCore
