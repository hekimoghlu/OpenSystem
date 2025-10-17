/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "SVGGradientElement.h"
#include "SVGUnitTypes.h"

namespace WebCore {

struct GradientAttributes {
    GradientAttributes()
        : m_spreadMethod(SVGSpreadMethodPad)
        , m_gradientUnits(SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX)
        , m_spreadMethodSet(false)
        , m_gradientUnitsSet(false)
        , m_gradientTransformSet(false)
    {
    }

    SVGSpreadMethodType spreadMethod() const { return static_cast<SVGSpreadMethodType>(m_spreadMethod); }
    SVGUnitTypes::SVGUnitType gradientUnits() const { return static_cast<SVGUnitTypes::SVGUnitType>(m_gradientUnits); }
    AffineTransform gradientTransform() const { return m_gradientTransform; }
    const GradientColorStops& stops() const { return m_stops; }

    void setSpreadMethod(SVGSpreadMethodType value)
    {
        m_spreadMethod = value;
        m_spreadMethodSet = true;
    }

    void setGradientUnits(SVGUnitTypes::SVGUnitType unitType)
    {
        m_gradientUnits = unitType;
        m_gradientUnitsSet = true;
    }

    void setGradientTransform(const AffineTransform& value)
    {
        m_gradientTransform = value;
        m_gradientTransformSet = true;
    }

    void setStops(GradientColorStops&& value)
    {
        m_stops = WTFMove(value);
    }

    bool hasSpreadMethod() const { return m_spreadMethodSet; }
    bool hasGradientUnits() const { return m_gradientUnitsSet; }
    bool hasGradientTransform() const { return m_gradientTransformSet; }
    bool hasStops() const { return !m_stops.isEmpty(); }

private:
    // Properties
    AffineTransform m_gradientTransform;
    GradientColorStops m_stops;

    unsigned m_spreadMethod : 2;
    unsigned m_gradientUnits : 2;

    // Property states
    unsigned m_spreadMethodSet : 1;
    unsigned m_gradientUnitsSet : 1;
    unsigned m_gradientTransformSet : 1;
};

struct SameSizeAsGradientAttributes {
    AffineTransform a;
    GradientColorStops b;
    unsigned c : 7;
};

static_assert(sizeof(GradientAttributes) == sizeof(SameSizeAsGradientAttributes));

} // namespace WebCore
