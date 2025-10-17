/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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

#include <wtf/Forward.h>
#if HAVE(IOSURFACE)
#include "IOSurface.h"
#endif
#include "PixelFormat.h"

namespace WebCore {
enum class ImageBufferPixelFormat : uint8_t {
    BGRX8,
    BGRA8,
#if HAVE(IOSURFACE_RGB10)
    RGB10,
    RGB10A8,
#endif
#if HAVE(HDR_SUPPORT)
    RGBA16F,
#endif
};

constexpr PixelFormat convertToPixelFormat(ImageBufferPixelFormat format)
{
    switch (format) {
    case ImageBufferPixelFormat::BGRX8:
        return PixelFormat::BGRX8;
    case ImageBufferPixelFormat::BGRA8:
        return PixelFormat::BGRA8;
#if HAVE(IOSURFACE_RGB10)
    case ImageBufferPixelFormat::RGB10:
        return PixelFormat::RGB10;
    case ImageBufferPixelFormat::RGB10A8:
        return PixelFormat::RGB10A8;
#endif
#if HAVE(HDR_SUPPORT)
    case ImageBufferPixelFormat::RGBA16F:
        return PixelFormat::RGBA16F;
#endif
    }

    ASSERT_NOT_REACHED();
    return PixelFormat::BGRX8;
}

#if HAVE(IOSURFACE)
constexpr IOSurface::Format convertToIOSurfaceFormat(ImageBufferPixelFormat format)
{
    switch (format) {
    case ImageBufferPixelFormat::BGRX8:
        return IOSurface::Format::BGRX;
    case ImageBufferPixelFormat::BGRA8:
        return IOSurface::Format::BGRA;
#if HAVE(IOSURFACE_RGB10)
    case ImageBufferPixelFormat::RGB10:
        return IOSurface::Format::RGB10;
    case ImageBufferPixelFormat::RGB10A8:
        return IOSurface::Format::RGB10A8;
#endif
#if HAVE(HDR_SUPPORT)
    case ImageBufferPixelFormat::RGBA16F:
        return IOSurface::Format::RGBA16F;
#endif
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return IOSurface::Format::BGRA;
    }
}
#endif

} // namespace WebCore
