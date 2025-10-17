/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "ImageBufferUtilitiesSkia.h"

#if USE(SKIA)
#include "GLContext.h"
#include "MIMETypeRegistry.h"
#include "PlatformDisplay.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkData.h>
#include <skia/core/SkImage.h>
#include <skia/core/SkStream.h>
#include <skia/encode/SkJpegEncoder.h>
#include <skia/encode/SkPngEncoder.h>
#include <skia/encode/SkWebpEncoder.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/text/WTFString.h>

namespace WebCore {

class VectorSkiaWritableStream final : public SkWStream {
public:
    explicit VectorSkiaWritableStream(Vector<uint8_t>& vector)
        : m_vector(vector)
    {
    }

    bool write(const void* data, size_t length) override
    {
        m_vector.append(unsafeMakeSpan(static_cast<const uint8_t*>(data), length));
        return true;
    }

    void flush() override { }

    size_t bytesWritten() const override { return m_vector.size(); }

private:
    Vector<uint8_t>& m_vector;
};

static sk_sp<SkData> encodeAcceleratedImage(SkImage* image, const String& mimeType, std::optional<double> quality)
{
    if (!PlatformDisplay::sharedDisplay().skiaGLContext()->makeContextCurrent())
        return nullptr;

    GrDirectContext* grContext = PlatformDisplay::sharedDisplay().skiaGrContext();

    if (MIMETypeRegistry::isJPEGMIMEType(mimeType)) {
        SkJpegEncoder::Options options;
        if (quality && *quality >= 0.0 && *quality <= 1.0)
            options.fQuality = static_cast<int>(*quality * 100.0 + 0.5);
        return SkJpegEncoder::Encode(grContext, image, options);
    }

    if (equalLettersIgnoringASCIICase(mimeType, "image/webp"_s)) {
        SkWebpEncoder::Options options;
        if (quality && *quality >= 0.0 && *quality <= 1.0)
            options.fQuality = static_cast<int>(*quality * 100.0 + 0.5);
        return SkWebpEncoder::Encode(grContext, image, options);
    }

    if (equalLettersIgnoringASCIICase(mimeType, "image/png"_s))
        return SkPngEncoder::Encode(grContext, image, { });

    return nullptr;
}

static Vector<uint8_t> encodeUnacceleratedImage(const SkPixmap& pixmap, const String& mimeType, std::optional<double> quality)
{
    Vector<uint8_t> result;
    VectorSkiaWritableStream stream(result);

    if (MIMETypeRegistry::isJPEGMIMEType(mimeType)) {
        SkJpegEncoder::Options options;
        if (quality && *quality >= 0.0 && *quality <= 1.0)
            options.fQuality = static_cast<int>(*quality * 100.0 + 0.5);
        if (!SkJpegEncoder::Encode(&stream, pixmap, options))
            return { };
    } else if (equalLettersIgnoringASCIICase(mimeType, "image/webp"_s)) {
        SkWebpEncoder::Options options;
        if (quality && *quality >= 0.0 && *quality <= 1.0)
            options.fQuality = static_cast<int>(*quality * 100.0 + 0.5);
        if (!SkWebpEncoder::Encode(&stream, pixmap, options))
            return { };
    } else if (equalLettersIgnoringASCIICase(mimeType, "image/png"_s)) {
        if (!SkPngEncoder::Encode(&stream, pixmap, { }))
            return { };
    }

    return result;
}

Vector<uint8_t> encodeData(SkImage* image, const String& mimeType, std::optional<double> quality)
{
    if (image->isTextureBacked()) {
        auto data = encodeAcceleratedImage(image, mimeType, quality);
        if (!data)
            return { };

        return unsafeMakeSpan(reinterpret_cast<const uint8_t*>(data->data()), data->size());
    }

    SkPixmap pixmap;
    if (!image->peekPixels(&pixmap))
        return { };

    return encodeUnacceleratedImage(pixmap, mimeType, quality);
}

} // namespace WebCore

#endif // USE(SKIA)
