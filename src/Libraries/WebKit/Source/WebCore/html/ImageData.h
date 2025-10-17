/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

#include "ByteArrayPixelBuffer.h"
#include "ExceptionOr.h"
#include "ImageDataArray.h"
#include "ImageDataSettings.h"
#include "IntSize.h"
#include "PredefinedColorSpace.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/Forward.h>

namespace WebCore {

class ImageData : public RefCounted<ImageData> {
public:
    WEBCORE_EXPORT static Ref<ImageData> create(Ref<ByteArrayPixelBuffer>&&);
    WEBCORE_EXPORT static RefPtr<ImageData> create(RefPtr<ByteArrayPixelBuffer>&&);
    WEBCORE_EXPORT static RefPtr<ImageData> create(const IntSize&, PredefinedColorSpace);
    WEBCORE_EXPORT static RefPtr<ImageData> create(const IntSize&, ImageDataArray&&, PredefinedColorSpace);

    WEBCORE_EXPORT static ExceptionOr<Ref<ImageData>> create(unsigned sw, unsigned sh, PredefinedColorSpace defaultColorSpace, std::optional<ImageDataSettings> = std::nullopt, std::span<const uint8_t> = { });
    WEBCORE_EXPORT static ExceptionOr<Ref<ImageData>> create(unsigned sw, unsigned sh, std::optional<ImageDataSettings>);
    WEBCORE_EXPORT static ExceptionOr<Ref<ImageData>> create(ImageDataArray&&, unsigned sw, std::optional<unsigned> sh, std::optional<ImageDataSettings>);

    WEBCORE_EXPORT ~ImageData();

    static PredefinedColorSpace computeColorSpace(std::optional<ImageDataSettings>, PredefinedColorSpace defaultColorSpace = PredefinedColorSpace::SRGB);

    const IntSize& size() const { return m_size; }

    int width() const { return m_size.width(); }
    int height() const { return m_size.height(); }
    const ImageDataArray& data() const { return m_data; }
    PredefinedColorSpace colorSpace() const { return m_colorSpace; }
    ImageDataStorageFormat storageFormat() const { return m_data.storageFormat(); }

    Ref<ByteArrayPixelBuffer> pixelBuffer() const;

private:
    explicit ImageData(const IntSize&, ImageDataArray&&, PredefinedColorSpace);

    IntSize m_size;
    ImageDataArray m_data;
    PredefinedColorSpace m_colorSpace;
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const ImageData&);

} // namespace WebCore
