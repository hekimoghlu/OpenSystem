/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

#include "ColorSpace.h"
#include <CoreGraphics/CoreGraphics.h>
#include <optional>
#include <wtf/cf/TypeCastsCF.h>

WTF_DECLARE_CF_TYPE_TRAIT(CGColorSpace);

namespace WebCore {

template<ColorSpace> struct CGColorSpaceMapping;

WEBCORE_EXPORT CGColorSpaceRef sRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::SRGB> { static CGColorSpaceRef colorSpace() { return sRGBColorSpaceRef(); } };

#if HAVE(CORE_GRAPHICS_ADOBE_RGB_1998_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef adobeRGB1998ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::A98RGB> { static CGColorSpaceRef colorSpace() { return adobeRGB1998ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::A98RGB> { };
#endif

#if HAVE(CORE_GRAPHICS_DISPLAY_P3_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef displayP3ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::DisplayP3> { static CGColorSpaceRef colorSpace() { return displayP3ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::DisplayP3> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ADOBE_RGB_1998_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedAdobeRGB1998ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedA98RGB> { static CGColorSpaceRef colorSpace() { return extendedAdobeRGB1998ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedA98RGB> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_DISPLAY_P3_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedDisplayP3ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedDisplayP3> { static CGColorSpaceRef colorSpace() { return extendedDisplayP3ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedDisplayP3> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ITUR_2020_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedITUR_2020ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedRec2020> { static CGColorSpaceRef colorSpace() { return extendedITUR_2020ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedRec2020> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_LINEAR_SRGB_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedLinearSRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedLinearSRGB> { static CGColorSpaceRef colorSpace() { return extendedLinearSRGBColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedLinearSRGB> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_ROMMRGB_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedROMMRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedProPhotoRGB> { static CGColorSpaceRef colorSpace() { return extendedROMMRGBColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedProPhotoRGB> { };
#endif

#if HAVE(CORE_GRAPHICS_EXTENDED_SRGB_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef extendedSRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedSRGB> { static CGColorSpaceRef colorSpace() { return extendedSRGBColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ExtendedSRGB> { };
#endif

#if HAVE(CORE_GRAPHICS_ITUR_2020_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef ITUR_2020ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::Rec2020> { static CGColorSpaceRef colorSpace() { return ITUR_2020ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::Rec2020> { };
#endif

#if HAVE(CORE_GRAPHICS_LINEAR_SRGB_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef linearSRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::LinearSRGB> { static CGColorSpaceRef colorSpace() { return linearSRGBColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::LinearSRGB> { };
#endif

#if HAVE(CORE_GRAPHICS_ROMMRGB_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef ROMMRGBColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::ProPhotoRGB> { static CGColorSpaceRef colorSpace() { return ROMMRGBColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::ProPhotoRGB> { };
#endif

#if HAVE(CORE_GRAPHICS_XYZ_D50_COLOR_SPACE)
WEBCORE_EXPORT CGColorSpaceRef xyzD50ColorSpaceRef();
template<> struct CGColorSpaceMapping<ColorSpace::XYZ_D50> { static CGColorSpaceRef colorSpace() { return xyzD50ColorSpaceRef(); } };
#else
template<> struct CGColorSpaceMapping<ColorSpace::XYZ_D50> { };
#endif

// FIXME: Add support for these once/if CoreGraphics adds support for them.
template<> struct CGColorSpaceMapping<ColorSpace::HSL> { };
template<> struct CGColorSpaceMapping<ColorSpace::HWB> { };
template<> struct CGColorSpaceMapping<ColorSpace::LCH> { };
template<> struct CGColorSpaceMapping<ColorSpace::Lab> { };
template<> struct CGColorSpaceMapping<ColorSpace::OKLab> { };
template<> struct CGColorSpaceMapping<ColorSpace::OKLCH> { };
template<> struct CGColorSpaceMapping<ColorSpace::XYZ_D65> { };


WEBCORE_EXPORT std::optional<ColorSpace> colorSpaceForCGColorSpace(CGColorSpaceRef);


template<ColorSpace, typename = void> inline constexpr bool HasCGColorSpaceMapping = false;
template<ColorSpace space> inline constexpr bool HasCGColorSpaceMapping<space, std::void_t<decltype(CGColorSpaceMapping<space>::colorSpace())>> = true;
static_assert(HasCGColorSpaceMapping<ColorSpace::SRGB>, "An SRGB color space mapping must be supported on all platforms.");

template<ColorSpace space, bool = HasCGColorSpaceMapping<space>> struct CGColorSpaceMappingOrNullGetter { static CGColorSpaceRef colorSpace() { return nullptr; } };
template<ColorSpace space> struct CGColorSpaceMappingOrNullGetter<space, true> { static CGColorSpaceRef colorSpace() { return CGColorSpaceMapping<space>::colorSpace(); } };

template<ColorSpace space> CGColorSpaceRef cachedCGColorSpace()
{
    return CGColorSpaceMapping<space>::colorSpace();
}

template<ColorSpace space> CGColorSpaceRef cachedNullableCGColorSpace()
{
    return CGColorSpaceMappingOrNullGetter<space>::colorSpace();
}

inline CGColorSpaceRef cachedNullableCGColorSpace(ColorSpace colorSpace)
{
    switch (colorSpace) {
    case ColorSpace::A98RGB:
        return cachedNullableCGColorSpace<ColorSpace::A98RGB>();
    case ColorSpace::DisplayP3:
        return cachedNullableCGColorSpace<ColorSpace::DisplayP3>();
    case ColorSpace::ExtendedA98RGB:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedA98RGB>();
    case ColorSpace::ExtendedDisplayP3:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedDisplayP3>();
    case ColorSpace::ExtendedLinearSRGB:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedLinearSRGB>();
    case ColorSpace::ExtendedProPhotoRGB:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedProPhotoRGB>();
    case ColorSpace::ExtendedRec2020:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedRec2020>();
    case ColorSpace::ExtendedSRGB:
        return cachedNullableCGColorSpace<ColorSpace::ExtendedSRGB>();
    case ColorSpace::HSL:
        return cachedNullableCGColorSpace<ColorSpace::HSL>();
    case ColorSpace::HWB:
        return cachedNullableCGColorSpace<ColorSpace::HWB>();
    case ColorSpace::LCH:
        return cachedNullableCGColorSpace<ColorSpace::LCH>();
    case ColorSpace::Lab:
        return cachedNullableCGColorSpace<ColorSpace::Lab>();
    case ColorSpace::LinearSRGB:
        return cachedNullableCGColorSpace<ColorSpace::LinearSRGB>();
    case ColorSpace::OKLCH:
        return cachedNullableCGColorSpace<ColorSpace::OKLCH>();
    case ColorSpace::OKLab:
        return cachedNullableCGColorSpace<ColorSpace::OKLab>();
    case ColorSpace::ProPhotoRGB:
        return cachedNullableCGColorSpace<ColorSpace::ProPhotoRGB>();
    case ColorSpace::Rec2020:
        return cachedNullableCGColorSpace<ColorSpace::Rec2020>();
    case ColorSpace::SRGB:
        return cachedNullableCGColorSpace<ColorSpace::SRGB>();
    case ColorSpace::XYZ_D50:
        return cachedNullableCGColorSpace<ColorSpace::XYZ_D50>();
    case ColorSpace::XYZ_D65:
        return cachedNullableCGColorSpace<ColorSpace::XYZ_D65>();
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

}
