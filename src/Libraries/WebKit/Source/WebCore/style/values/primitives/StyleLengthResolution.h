/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

namespace WebCore {

class CSSToLengthConversionData;
class FontCascade;
class RenderStyle;
class RenderView;

enum CSSPropertyID : uint16_t;

namespace CSS {
enum class LengthUnit : uint8_t;
}

namespace Style {

// FIXME: These functions have odd names and invariants and could use improvements.

// NOTE: `computeUnzoomedNonCalcLengthDouble` has the following restrictions:
//
// It can never be called with the following LengthUnits:
//    Lh, Rlh, Cqw, Cqh, Cqi, Cqb, Cqmin, Cqmax (line height, and container-percentage units)
//
// If `fontCascadeForUnit` is nullptr, it additionally cannot be called with the following LengthUnits:
//    Em, QuirkyEm, Ex, Cap, Ch, Ic, Rca, Rc, Re, Re, Ri (font and root font dependent units)
//
// If `RenderView` is nullptr, the following LengthUnits will all cause a return value of zero:
//    Vw, Vh, Vmin, Vmax, Vb, Vi, Svw, Svh, Svmin, Svmax, Svb, Svi, Lvw, Lvh, Lvmin, Lvmax, Lvb, Lvi, Dvw, Dvh, Dvmin, Dvmax, Dvb, Dvi (viewport-percentage units)
double computeUnzoomedNonCalcLengthDouble(double value, CSS::LengthUnit, CSSPropertyID, const FontCascade* fontCascadeForUnit = nullptr, const RenderView* = nullptr);

double computeNonCalcLengthDouble(double value, CSS::LengthUnit, const CSSToLengthConversionData&);

// True if `computeNonCalcLengthDouble` would produce identical results when resolved against both these styles.
bool equalForLengthResolution(const RenderStyle&, const RenderStyle&);

} // namespace Style
} // namespace WebCore
