/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#include "SVGPropertyTraits.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class SVGUnitTypes final : public RefCounted<SVGUnitTypes> {
public:
    enum SVGUnitType : uint8_t {
        SVG_UNIT_TYPE_UNKNOWN               = 0,
        SVG_UNIT_TYPE_USERSPACEONUSE        = 1,
        SVG_UNIT_TYPE_OBJECTBOUNDINGBOX     = 2
    };

private:
    SVGUnitTypes() { }
};

template<>
struct SVGPropertyTraits<SVGUnitTypes::SVGUnitType> {
    static unsigned highestEnumValue() { return SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX; }

    static String toString(SVGUnitTypes::SVGUnitType type)
    {
        switch (type) {
        case SVGUnitTypes::SVG_UNIT_TYPE_UNKNOWN:
            return emptyString();
        case SVGUnitTypes::SVG_UNIT_TYPE_USERSPACEONUSE:
            return "userSpaceOnUse"_s;
        case SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX:
            return "objectBoundingBox"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static SVGUnitTypes::SVGUnitType fromString(const String& value)
    {
        if (value == "userSpaceOnUse"_s)
            return SVGUnitTypes::SVG_UNIT_TYPE_USERSPACEONUSE;
        if (value == "objectBoundingBox"_s)
            return SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX;
        return SVGUnitTypes::SVG_UNIT_TYPE_UNKNOWN;
    }
};

} // namespace WebCore
