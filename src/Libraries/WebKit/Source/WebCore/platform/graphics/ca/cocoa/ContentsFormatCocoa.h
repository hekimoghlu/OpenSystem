/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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

#if PLATFORM(COCOA)

#import "ContentsFormat.h"
#if HAVE(IOSURFACE)
#import "IOSurface.h"
#endif
#import <pal/spi/cocoa/QuartzCoreSPI.h>

namespace WebCore {

#if HAVE(IOSURFACE)
constexpr IOSurface::Format convertToIOSurfaceFormat(ContentsFormat contentsFormat)
{
    switch (contentsFormat) {
    case ContentsFormat::RGBA8:
        return IOSurface::Format::BGRA;
#if HAVE(IOSURFACE_RGB10)
    case ContentsFormat::RGBA10:
        return IOSurface::Format::RGB10;
#endif
#if HAVE(HDR_SUPPORT)
    case ContentsFormat::RGBA16F:
        return IOSurface::Format::RGBA16F;
#endif
    }
}
#endif

constexpr NSString *contentsFormatString(ContentsFormat contentsFormat)
{
    switch (contentsFormat) {
    case ContentsFormat::RGBA8:
        return nil;
#if HAVE(IOSURFACE_RGB10)
    case ContentsFormat::RGBA10:
        return kCAContentsFormatRGBA10XR;
#endif
#if HAVE(HDR_SUPPORT)
    case ContentsFormat::RGBA16F:
        return kCAContentsFormatRGBA16Float;
#endif
    }
}

} // namespace WebCore

#endif // PLATFORM(COCOA)
