/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"

#include "CSSPropertyNames.h"
#include "ComputedStyleDependencies.h"

namespace WebCore {
namespace CSS {

// MARK: - Computed Style Dependencies

void ComputedStyleDependenciesCollector<LengthUnit>::operator()(ComputedStyleDependencies& dependencies, LengthUnit lengthUnit)
{
    using enum LengthUnit;

    switch (lengthUnit) {
    case Rcap:
    case Rch:
    case Rex:
    case Ric:
    case Rem:
        dependencies.rootProperties.appendIfNotContains(CSSPropertyFontSize);
        break;
    case Rlh:
        dependencies.rootProperties.appendIfNotContains(CSSPropertyFontSize);
        dependencies.rootProperties.appendIfNotContains(CSSPropertyLineHeight);
        break;
    case Em:
    case QuirkyEm:
    case Ex:
    case Cap:
    case Ch:
    case Ic:
        dependencies.properties.appendIfNotContains(CSSPropertyFontSize);
        break;
    case Lh:
        dependencies.properties.appendIfNotContains(CSSPropertyFontSize);
        dependencies.properties.appendIfNotContains(CSSPropertyLineHeight);
        break;
    case Cqw:
    case Cqh:
    case Cqi:
    case Cqb:
    case Cqmin:
    case Cqmax:
        dependencies.containerDimensions = true;
        break;
    case Vw:
    case Vh:
    case Vmin:
    case Vmax:
    case Vb:
    case Vi:
    case Svw:
    case Svh:
    case Svmin:
    case Svmax:
    case Svb:
    case Svi:
    case Lvw:
    case Lvh:
    case Lvmin:
    case Lvmax:
    case Lvb:
    case Lvi:
    case Dvw:
    case Dvh:
    case Dvmin:
    case Dvmax:
    case Dvb:
    case Dvi:
        dependencies.viewportDimensions = true;
        break;
    case Px:
    case Cm:
    case Mm:
    case In:
    case Pt:
    case Pc:
    case Q:
        break;
    }
}

} // namespace CSS
} // namespace WebCore
