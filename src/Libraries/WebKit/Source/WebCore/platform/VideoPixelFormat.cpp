/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
#include "VideoPixelFormat.h"

#if USE(GSTREAMER)
#include <gst/video/video-format.h>
#endif

#if PLATFORM(COCOA)
#include <pal/cf/CoreMediaSoftLink.h>
#include "CoreVideoSoftLink.h"
#endif

namespace WebCore {

VideoPixelFormat convertVideoFramePixelFormat(uint32_t format, bool shouldDiscardAlpha)
{
#if PLATFORM(COCOA)
    if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || format == kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarVideoRange)
        return VideoPixelFormat::NV12;
    if (format == kCVPixelFormatType_32BGRA)
        return shouldDiscardAlpha ? VideoPixelFormat::BGRX : VideoPixelFormat::BGRA;
    if (format == kCVPixelFormatType_32ARGB)
        return shouldDiscardAlpha ? VideoPixelFormat::RGBX : VideoPixelFormat::RGBA;
#elif USE(GSTREAMER)
    switch (format) {
    case GST_VIDEO_FORMAT_I420:
        return VideoPixelFormat::I420;
    case GST_VIDEO_FORMAT_A420:
        return VideoPixelFormat::I420A;
    case GST_VIDEO_FORMAT_Y42B:
        return VideoPixelFormat::I422;
    case GST_VIDEO_FORMAT_Y444:
        return VideoPixelFormat::I444;
    case GST_VIDEO_FORMAT_NV12:
        return VideoPixelFormat::NV12;
    case GST_VIDEO_FORMAT_RGBA:
        return shouldDiscardAlpha ? VideoPixelFormat::RGBX : VideoPixelFormat::RGBA;
    case GST_VIDEO_FORMAT_RGBx:
        return VideoPixelFormat::RGBX;
    case GST_VIDEO_FORMAT_BGRA:
        return shouldDiscardAlpha ? VideoPixelFormat::BGRX : VideoPixelFormat::BGRA;
    case GST_VIDEO_FORMAT_ARGB:
        return shouldDiscardAlpha ? VideoPixelFormat::RGBX : VideoPixelFormat::RGBA;
    case GST_VIDEO_FORMAT_BGRx:
        return VideoPixelFormat::BGRX;
    default:
        break;
    }
#else
    UNUSED_PARAM(format);
    UNUSED_PARAM(shouldDiscardAlpha);
#endif
    return VideoPixelFormat::I420;
}

} // namespace WebCore

