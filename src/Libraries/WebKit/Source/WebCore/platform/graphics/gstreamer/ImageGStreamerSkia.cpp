/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#include "ImageGStreamer.h"

#if ENABLE(VIDEO) && USE(GSTREAMER) && USE(SKIA)

#include "NotImplemented.h"
#include <skia/ColorSpaceSkia.h>
#include <skia/core/SkImage.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkPixmap.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

ImageGStreamer::ImageGStreamer(GRefPtr<GstSample>&& sample)
    : m_sample(WTFMove(sample))
{
    GstBuffer* buffer = gst_sample_get_buffer(m_sample.get());
    if (UNLIKELY(!GST_IS_BUFFER(buffer)))
        return;

    GstMappedFrame videoFrame(m_sample, GST_MAP_READ);
    if (!videoFrame)
        return;

    auto* videoInfo = videoFrame.info();

    // The frame has to be RGB so we can paint it.
    ASSERT(GST_VIDEO_INFO_IS_RGB(videoInfo));

    // The video buffer may have these formats in these cases:
    // { BGRx, BGRA } on little endian
    // { xRGB, ARGB } on big endian:
    // { RGBx, RGBA }
    SkColorType colorType = kUnknown_SkColorType;
    SkAlphaType alphaType = kUnknown_SkAlphaType;
    switch (videoFrame.format()) {
    case GST_VIDEO_FORMAT_BGRx:
        colorType = kBGRA_8888_SkColorType;
        alphaType = kOpaque_SkAlphaType;
        break;
    case GST_VIDEO_FORMAT_BGRA:
        colorType = kBGRA_8888_SkColorType;
        alphaType = kUnpremul_SkAlphaType;
        break;
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_ARGB:
        // FIXME: we need a conversion here.
        notImplemented();
        return;
    case GST_VIDEO_FORMAT_RGBx:
        colorType = kRGB_888x_SkColorType;
        alphaType = kOpaque_SkAlphaType;
        break;
    case GST_VIDEO_FORMAT_RGBA:
        colorType = kRGBA_8888_SkColorType;
        alphaType = kUnpremul_SkAlphaType;
        break;
    default:
        ASSERT_NOT_REACHED();
        break;
    }

    m_size = { static_cast<float>(videoFrame.width()), static_cast<float>(videoFrame.height()) };

    auto toSkiaColorSpace = [](const PlatformVideoColorSpace& videoColorSpace) {
        // Only valid, full-range RGB spaces are supported.
        if (!videoColorSpace.primaries || !videoColorSpace.transfer || !videoColorSpace.matrix || !videoColorSpace.fullRange)
            return sk_sp<SkColorSpace>();
        const auto& matrix = *videoColorSpace.matrix;
        if (matrix != PlatformVideoMatrixCoefficients::Rgb || !*videoColorSpace.fullRange)
            return sk_sp<SkColorSpace>();

        const auto& primaries = *videoColorSpace.primaries;
        const auto& transfer = *videoColorSpace.transfer;
        if (primaries == PlatformVideoColorPrimaries::Bt709) {
            if (transfer == PlatformVideoTransferCharacteristics::Iec6196621)
                return SkColorSpace::MakeSRGB();
            if (transfer == PlatformVideoTransferCharacteristics::Linear)
                return SkColorSpace::MakeSRGBLinear();
        }

        skcms_TransferFunction transferFunction = SkNamedTransferFn::kSRGB;
        switch (transfer) {
        case PlatformVideoTransferCharacteristics::Iec6196621:
            break;
        case PlatformVideoTransferCharacteristics::Linear:
            transferFunction = SkNamedTransferFn::kLinear;
            break;
        case PlatformVideoTransferCharacteristics::Bt2020_10bit:
        case PlatformVideoTransferCharacteristics::Bt2020_12bit:
            transferFunction = SkNamedTransferFn::kRec2020;
            break;
        case PlatformVideoTransferCharacteristics::PQ:
            transferFunction = SkNamedTransferFn::kPQ;
            break;
        case PlatformVideoTransferCharacteristics::HLG:
            transferFunction = SkNamedTransferFn::kHLG;
            break;
        default:
            // No known conversion to skia's skcms_TransferFunction - falling back to kSRGB.
            break;
        }

        skcms_Matrix3x3 gamut = SkNamedGamut::kSRGB;
        switch (primaries) {
        case PlatformVideoColorPrimaries::Bt709:
            break;
        case PlatformVideoColorPrimaries::Bt2020:
            gamut = SkNamedGamut::kRec2020;
            break;
        case PlatformVideoColorPrimaries::Smpte432:
            gamut = SkNamedGamut::kDisplayP3;
            break;
        default:
            // No known conversion to skia's skcms_Matrix3x3 - falling back to kSRGB.
            break;
        }

        return SkColorSpace::MakeRGB(transferFunction, gamut);
    };
    auto imageInfo = SkImageInfo::Make(videoFrame.width(), videoFrame.height(), colorType, alphaType, toSkiaColorSpace(videoColorSpaceFromInfo(*videoInfo)));

    // Copy the buffer data. Keeping the whole mapped GstVideoFrame alive would increase memory
    // pressure and the file descriptor(s) associated with the buffer pool open. We only need the
    // data here.
    SkPixmap pixmap(imageInfo, videoFrame.planeData(0), videoFrame.planeStride(0));
    m_image = SkImages::RasterFromPixmapCopy(pixmap);

    if (auto* cropMeta = gst_buffer_get_video_crop_meta(buffer))
        m_cropRect = FloatRect(cropMeta->x, cropMeta->y, cropMeta->width, cropMeta->height);
}

ImageGStreamer::~ImageGStreamer() = default;

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER) && USE(SKIA)
