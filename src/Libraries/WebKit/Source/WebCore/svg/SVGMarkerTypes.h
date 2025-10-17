/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#include "CommonAtomStrings.h"
#include "SVGAngleValue.h"
#include "SVGPropertyTraits.h"

namespace WebCore {

enum SVGMarkerUnitsType {
    SVGMarkerUnitsUnknown = 0,
    SVGMarkerUnitsUserSpaceOnUse,
    SVGMarkerUnitsStrokeWidth
};

enum SVGMarkerOrientType {
    SVGMarkerOrientUnknown = 0,
    SVGMarkerOrientAuto,
    SVGMarkerOrientAngle,

    // The DOM can't set the property 'orientType' to this value. It is used only
    // internally when setting the 'orient' attribute to "auto-start-reverse".
    SVGMarkerOrientAutoStartReverse = SVGMarkerOrientUnknown
};
    
template<>
struct SVGPropertyTraits<SVGMarkerUnitsType> {
    static unsigned highestEnumValue() { return SVGMarkerUnitsStrokeWidth; }
    static String toString(SVGMarkerUnitsType type)
    {
        switch (type) {
        case SVGMarkerUnitsUnknown:
            return emptyString();
        case SVGMarkerUnitsUserSpaceOnUse:
            return "userSpaceOnUse"_s;
        case SVGMarkerUnitsStrokeWidth:
            return "strokeWidth"_s;
        }
        
        ASSERT_NOT_REACHED();
        return emptyString();
    }
    static SVGMarkerUnitsType fromString(const String& value)
    {
        if (value == "userSpaceOnUse"_s)
            return SVGMarkerUnitsUserSpaceOnUse;
        if (value == "strokeWidth"_s)
            return SVGMarkerUnitsStrokeWidth;
        return SVGMarkerUnitsUnknown;
    }
};

template<>
struct SVGPropertyTraits<SVGMarkerOrientType> {
    static const String autoStartReverseString()
    {
        static const NeverDestroyed<String> autoStartReverseString = MAKE_STATIC_STRING_IMPL("auto-start-reverse");
        return autoStartReverseString;
    }
    static unsigned highestEnumValue() { return SVGMarkerOrientAngle; }
    static SVGMarkerOrientType fromString(const String& string)
    {
        if (string == autoAtom())
            return SVGMarkerOrientAuto;
        if (string == autoStartReverseString())
            return SVGMarkerOrientAutoStartReverse;
        return SVGMarkerOrientUnknown;
    }
    static String toString(SVGMarkerOrientType type)
    {
        if (type == SVGMarkerOrientAuto)
            return autoAtom();
        if (type == SVGMarkerOrientAutoStartReverse)
            return autoStartReverseString();
        return emptyString();
    }
};

template<>
struct SVGPropertyTraits<std::pair<SVGAngleValue, SVGMarkerOrientType>> {
    static std::pair<SVGAngleValue, SVGMarkerOrientType> fromString(const String& string)
    {
        SVGAngleValue angle;
        SVGMarkerOrientType orientType = SVGPropertyTraits<SVGMarkerOrientType>::fromString(string);
        if (orientType == SVGMarkerOrientUnknown) {
            auto result = angle.setValueAsString(string);
            if (!result.hasException())
                orientType = SVGMarkerOrientAngle;
        }
        return std::make_pair(angle, orientType);
    }
};

} // namespace WebCore
