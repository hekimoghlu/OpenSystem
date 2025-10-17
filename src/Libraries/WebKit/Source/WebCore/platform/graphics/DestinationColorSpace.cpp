/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include "DestinationColorSpace.h"
#include "NotImplemented.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/text/TextStream.h>

#if USE(CG)
#include "ColorSpaceCG.h"
#include <pal/spi/cg/CoreGraphicsSPI.h>
#elif USE(SKIA)
#include "ColorSpaceSkia.h"
#endif

namespace WebCore {

#if USE(CG) || USE(SKIA)
#if USE(CG)
using KnownColorSpaceAccessor = CGColorSpaceRef();
#elif USE(SKIA)
using KnownColorSpaceAccessor = sk_sp<SkColorSpace>();
#endif
template<KnownColorSpaceAccessor accessor> static const DestinationColorSpace& knownColorSpace()
{
    static LazyNeverDestroyed<DestinationColorSpace> colorSpace;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        colorSpace.construct(accessor());
    });
    return colorSpace.get();
}
#else
template<PlatformColorSpace::Name name> static const DestinationColorSpace& knownColorSpace()
{
    static NeverDestroyed<DestinationColorSpace> colorSpace { name };
    return colorSpace.get();
}
#endif

const DestinationColorSpace& DestinationColorSpace::SRGB()
{
#if USE(CG) || USE(SKIA)
    return knownColorSpace<sRGBColorSpaceRef>();
#else
    return knownColorSpace<PlatformColorSpace::Name::SRGB>();
#endif
}

const DestinationColorSpace& DestinationColorSpace::LinearSRGB()
{
#if USE(CG) || USE(SKIA)
    return knownColorSpace<linearSRGBColorSpaceRef>();
#else
    return knownColorSpace<PlatformColorSpace::Name::LinearSRGB>();
#endif
}

#if ENABLE(DESTINATION_COLOR_SPACE_DISPLAY_P3)
const DestinationColorSpace& DestinationColorSpace::DisplayP3()
{
#if USE(CG) || USE(SKIA)
    return knownColorSpace<displayP3ColorSpaceRef>();
#else
    return knownColorSpace<PlatformColorSpace::Name::DisplayP3>();
#endif
}
#endif

bool operator==(const DestinationColorSpace& a, const DestinationColorSpace& b)
{
#if USE(CG)
    return CGColorSpaceEqualToColorSpace(a.platformColorSpace(), b.platformColorSpace());
#elif USE(SKIA)
    return SkColorSpace::Equals(a.platformColorSpace().get(), b.platformColorSpace().get());
#else
    return a.platformColorSpace() == b.platformColorSpace();
#endif
}

std::optional<DestinationColorSpace> DestinationColorSpace::asRGB() const
{
#if USE(CG)
    CGColorSpaceRef colorSpace = platformColorSpace();
    if (CGColorSpaceGetModel(colorSpace) == kCGColorSpaceModelIndexed)
        colorSpace = CGColorSpaceGetBaseColorSpace(colorSpace);

    if (CGColorSpaceGetModel(colorSpace) != kCGColorSpaceModelRGB)
        return std::nullopt;

    if (usesExtendedRange())
        return std::nullopt;

    return DestinationColorSpace(colorSpace);

#elif USE(SKIA)
    // When using skia, we're not using color spaces consisting of custom lookup tables, so we either yield SRGB or nothing.
    if (platformColorSpace()->isSRGB())
        return SRGB();
    return std::nullopt;

#else
    return *this;
#endif
}

bool DestinationColorSpace::supportsOutput() const
{
#if USE(CG)
    return CGColorSpaceSupportsOutput(platformColorSpace());
#else
    notImplemented();
    return true;
#endif
}

bool DestinationColorSpace::usesExtendedRange() const
{
#if USE(CG)
    return CGColorSpaceUsesExtendedRange(platformColorSpace());
#else
    notImplemented();
    return false;
#endif
}

TextStream& operator<<(TextStream& ts, const DestinationColorSpace& colorSpace)
{
    if (colorSpace == DestinationColorSpace::SRGB())
        ts << "sRGB";
    else if (colorSpace == DestinationColorSpace::LinearSRGB())
        ts << "LinearSRGB";
#if ENABLE(DESTINATION_COLOR_SPACE_DISPLAY_P3)
    else if (colorSpace == DestinationColorSpace::DisplayP3())
        ts << "DisplayP3";
#endif
#if USE(CG)
    else if (auto description = adoptCF(CGColorSpaceCopyICCProfileDescription(colorSpace.platformColorSpace())))
        ts << String(description.get());
#endif

    return ts;
}

}
