/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

namespace WTF {
class TextStream;
}

namespace WebCore {

namespace CSS {

// We always assume 96 CSS pixels in a CSS inch. This is the cold hard truth of the Web.
// At high DPI, we may scale a CSS pixel, but the ratio of the CSS pixel to the so-called
// "absolute" CSS length units like inch and pt is always fixed and never changes.
constexpr double pixelsPerInch = 96;

constexpr double pointsPerInch = 72;
constexpr double picasPerInch = 6;
constexpr double mmPerInch = 25.4;
constexpr double cmPerInch = 2.54;
constexpr double QPerInch = 25.4 * 4.0;

constexpr double pixelsPerCm = pixelsPerInch / cmPerInch;
constexpr double pixelsPerMm = pixelsPerInch / mmPerInch;
constexpr double pixelsPerQ = pixelsPerInch / QPerInch;
constexpr double pixelsPerPt = pixelsPerInch / pointsPerInch;
constexpr double pixelsPerPc = pixelsPerInch / picasPerInch;
constexpr double dppxPerX = 1.0;
constexpr double dppxPerDpi = 1.0 / pixelsPerInch;
constexpr double dppxPerDpcm = cmPerInch / pixelsPerInch;
constexpr double secondsPerMillisecond = 1.0 / 1000.0;
constexpr double hertzPerKilohertz = 1000.0;

}

// FIXME: No need to use all capitals and a CSS prefix on all these names. Should fix that.
enum class CSSUnitType : uint8_t {
    CSS_UNKNOWN,
    CSS_NUMBER,
    CSS_INTEGER,
    CSS_PERCENTAGE,
    CSS_EM,
    CSS_EX,
    CSS_PX,
    CSS_CM,
    CSS_MM,
    CSS_IN,
    CSS_PT,
    CSS_PC,
    CSS_DEG,
    CSS_RAD,
    CSS_GRAD,
    CSS_MS,
    CSS_S,
    CSS_HZ,
    CSS_KHZ,
    CSS_DIMENSION,
    CSS_STRING,
    CSS_URI,
    CSS_IDENT,
    CSS_ATTR,

    CSS_VW,
    CSS_VH,
    CSS_VMIN,
    CSS_VMAX,
    CSS_VB,
    CSS_VI,
    CSS_SVW,
    CSS_SVH,
    CSS_SVMIN,
    CSS_SVMAX,
    CSS_SVB,
    CSS_SVI,
    CSS_LVW,
    CSS_LVH,
    CSS_LVMIN,
    CSS_LVMAX,
    CSS_LVB,
    CSS_LVI,
    CSS_DVW,
    CSS_DVH,
    CSS_DVMIN,
    CSS_DVMAX,
    CSS_DVB,
    CSS_DVI,
    FirstViewportCSSUnitType = CSS_VW,
    LastViewportCSSUnitType = CSS_DVI,

    CSS_CQW,
    CSS_CQH,
    CSS_CQI,
    CSS_CQB,
    CSS_CQMIN,
    CSS_CQMAX,

    CSS_DPPX,
    CSS_X,
    CSS_DPI,
    CSS_DPCM,
    CSS_FR,
    CSS_Q,
    CSS_LH,
    CSS_RLH,

    CustomIdent,

    CSS_TURN,
    CSS_REM,
    CSS_REX,
    CSS_CAP,
    CSS_RCAP,
    CSS_CH,
    CSS_RCH,
    CSS_IC,
    CSS_RIC,

    CSS_CALC,
    CSS_CALC_PERCENTAGE_WITH_ANGLE,
    CSS_CALC_PERCENTAGE_WITH_LENGTH,

    CSS_FONT_FAMILY,

    CSS_PROPERTY_ID,
    CSS_VALUE_ID,
    
    // This value is used to handle quirky margins in reflow roots (body, td, and th) like WinIE.
    // The basic idea is that a stylesheet can use the value __qem (for quirky em) instead of em.
    // When the quirky value is used, if you're in quirks mode, the margin will collapse away
    // inside a table cell. This quirk is specified in the HTML spec but our impl is different.
    CSS_QUIRKY_EM

    // Note that CSSValue allocates 7 bits for m_primitiveUnitType, so there can be no value here > 127.
};

enum class CSSUnitCategory : uint8_t {
    Number,
    Percent,
    AbsoluteLength,
    FontRelativeLength,
    ViewportPercentageLength,
    Angle,
    Time,
    Frequency,
    Resolution,
    Flex,
    Other
};

CSSUnitCategory unitCategory(CSSUnitType);
CSSUnitType canonicalUnitTypeForCategory(CSSUnitCategory);
CSSUnitType canonicalUnitTypeForUnitType(CSSUnitType);
double conversionToCanonicalUnitsScaleFactor(CSSUnitType);
bool conversionToCanonicalUnitRequiresConversionData(CSSUnitType);

WTF::TextStream& operator<<(WTF::TextStream&, CSSUnitCategory);
WTF::TextStream& operator<<(WTF::TextStream&, CSSUnitType);

} // namespace WebCore
