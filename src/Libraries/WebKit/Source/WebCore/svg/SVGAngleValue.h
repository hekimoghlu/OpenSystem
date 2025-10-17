/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include "ExceptionOr.h"
#include "SVGPropertyTraits.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGAngleValue {
    WTF_MAKE_TZONE_ALLOCATED(SVGAngleValue);
public:
    enum Type {
        // FIXME: Change the casing style of the enum type or naming of the Angle Value (webkit.org/b/269429).
        SVG_ANGLETYPE_UNKNOWN = 0,
        SVG_ANGLETYPE_UNSPECIFIED = 1,
        SVG_ANGLETYPE_DEG = 2,
        SVG_ANGLETYPE_RAD = 3,
        SVG_ANGLETYPE_GRAD = 4,
        SVG_ANGLETYPE_TURN = 5
    };

    Type unitType() const { return m_unitType; }

    void setValue(float);
    float value() const;

    void setValueInSpecifiedUnits(float valueInSpecifiedUnits) { m_valueInSpecifiedUnits = valueInSpecifiedUnits; }
    float valueInSpecifiedUnits() const { return m_valueInSpecifiedUnits; }

    ExceptionOr<void> setValueAsString(const String&);
    String valueAsString() const;

    ExceptionOr<void> newValueSpecifiedUnits(unsigned short unitType, float valueInSpecifiedUnits);
    ExceptionOr<void> convertToSpecifiedUnits(unsigned short unitType);

private:
    Type m_unitType { SVG_ANGLETYPE_UNSPECIFIED };
    float m_valueInSpecifiedUnits { 0 };
};

}
