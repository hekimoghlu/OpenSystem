/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#include "ColorSpaceCG.h"

#if USE(CG)

#include <mutex>
#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RetainPtr.h>

namespace WebCore {

template<const CFStringRef& colorSpaceNameGlobalConstant> static CGColorSpaceRef namedColorSpace()
{
    static NeverDestroyed<RetainPtr<CGColorSpaceRef>> colorSpace;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        colorSpace.get() = adoptCF(CGColorSpaceCreateWithName(colorSpaceNameGlobalConstant));
        ASSERT(colorSpace.get());
    });
    return colorSpace.get().get();
}

#if HAVE(CORE_GRAPHICS_CREATE_EXTENDED_COLOR_SPACE)
template<const CFStringRef& colorSpaceNameGlobalConstant> static CGColorSpaceRef extendedNamedColorSpace()
{
    static NeverDestroyed<RetainPtr<CGColorSpaceRef>> colorSpace;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        colorSpace.get() = adoptCF(CGColorSpaceCreateExtended(namedColorSpace<colorSpaceNameGlobalConstant>()));
        ASSERT(colorSpace.get());
    });
    return colorSpace.get().get();
}
#endif

CGColorSpaceRef sRGBColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceSRGB>();
}

#if HAVE(CORE_GRAPHICS_ADOBE_RGB_1998_COLOR_SPACE)
CGColorSpaceRef adobeRGB1998ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceAdobeRGB1998>();
}
#endif

#if HAVE(CORE_GRAPHICS_DISPLAY_P3_COLOR_SPACE)
CGColorSpaceRef displayP3ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceDisplayP3>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ADOBE_RGB_1998_COLOR_SPACE)
CGColorSpaceRef extendedAdobeRGB1998ColorSpaceRef()
{
    return extendedNamedColorSpace<kCGColorSpaceAdobeRGB1998>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_DISPLAY_P3_COLOR_SPACE)
CGColorSpaceRef extendedDisplayP3ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceExtendedDisplayP3>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ITUR_2020_COLOR_SPACE)
CGColorSpaceRef extendedITUR_2020ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceExtendedITUR_2020>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_LINEAR_SRGB_COLOR_SPACE)
CGColorSpaceRef extendedLinearSRGBColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceExtendedLinearSRGB>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ROMMRGB_COLOR_SPACE)
CGColorSpaceRef extendedROMMRGBColorSpaceRef()
{
    return extendedNamedColorSpace<kCGColorSpaceROMMRGB>();
}
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_SRGB_COLOR_SPACE)
CGColorSpaceRef extendedSRGBColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceExtendedSRGB>();
}
#endif

#if HAVE(CORE_GRAPHICS_ITUR_2020_COLOR_SPACE)
CGColorSpaceRef ITUR_2020ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceITUR_2020>();
}
#endif

#if HAVE(CORE_GRAPHICS_LINEAR_SRGB_COLOR_SPACE)
CGColorSpaceRef linearSRGBColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceLinearSRGB>();
}
#endif

#if HAVE(CORE_GRAPHICS_ROMMRGB_COLOR_SPACE)
CGColorSpaceRef ROMMRGBColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceROMMRGB>();
}
#endif

#if HAVE(CORE_GRAPHICS_XYZ_D50_COLOR_SPACE)
CGColorSpaceRef xyzD50ColorSpaceRef()
{
    return namedColorSpace<kCGColorSpaceGenericXYZ>();
}

// FIXME: Figure out how to create a CoreGraphics XYZ-D65 color space and add a xyzD65ColorSpaceRef(). Perhaps CGColorSpaceCreateCalibratedRGB() with identify black point, D65 white point, and identity matrix.

#endif

std::optional<ColorSpace> colorSpaceForCGColorSpace(CGColorSpaceRef colorSpace)
{
    // First test for the four most common spaces, sRGB, Extended sRGB, DisplayP3 and Linear sRGB, and then test
    // the reset in alphabetical order.
    // FIXME: Consider using a UncheckedKeyHashMap (with CFHash based keys) rather than the linear set of tests.

    if (CGColorSpaceEqualToColorSpace(colorSpace, sRGBColorSpaceRef()))
        return ColorSpace::SRGB;

#if HAVE(CORE_GRAPHICS_EXTENDED_SRGB_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedSRGBColorSpaceRef()))
        return ColorSpace::ExtendedSRGB;
#endif

#if HAVE(CORE_GRAPHICS_DISPLAY_P3_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, displayP3ColorSpaceRef()))
        return ColorSpace::DisplayP3;
#endif

#if HAVE(CORE_GRAPHICS_LINEAR_SRGB_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, linearSRGBColorSpaceRef()))
        return ColorSpace::LinearSRGB;
#endif


#if HAVE(CORE_GRAPHICS_ADOBE_RGB_1998_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, adobeRGB1998ColorSpaceRef()))
        return ColorSpace::A98RGB;
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ADOBE_RGB_1998_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedAdobeRGB1998ColorSpaceRef()))
        return ColorSpace::ExtendedA98RGB;
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_DISPLAY_P3_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedDisplayP3ColorSpaceRef()))
        return ColorSpace::ExtendedDisplayP3;
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_LINEAR_SRGB_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedLinearSRGBColorSpaceRef()))
        return ColorSpace::ExtendedLinearSRGB;
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ITUR_2020_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedITUR_2020ColorSpaceRef()))
        return ColorSpace::ExtendedRec2020;
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ROMMRGB_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, extendedROMMRGBColorSpaceRef()))
        return ColorSpace::ExtendedProPhotoRGB;
#endif

#if HAVE(CORE_GRAPHICS_ITUR_2020_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, ITUR_2020ColorSpaceRef()))
        return ColorSpace::Rec2020;
#endif

#if HAVE(CORE_GRAPHICS_ROMMRGB_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, ROMMRGBColorSpaceRef()))
        return ColorSpace::ProPhotoRGB;
#endif

#if HAVE(CORE_GRAPHICS_XYZ_D50_COLOR_SPACE)
    if (CGColorSpaceEqualToColorSpace(colorSpace, xyzD50ColorSpaceRef()))
        return ColorSpace::XYZ_D50;
#endif

    // FIXME: Add support for remaining color spaces to support more direct conversions.

    return std::nullopt;
}

}

#endif // USE(CG)
