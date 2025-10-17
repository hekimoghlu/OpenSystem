/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#include "Length.h"

namespace WebCore {

// This class wraps the <track-breadth> which can be either a <percentage>, <length>, min-content, max-content
// or <flex>. This class avoids spreading the knowledge of <flex> throughout the rendering directory by adding
// an new unit to Length.h.
class GridLength {
public:
    GridLength(const Length& length)
        : m_length(length)
        , m_flex(0)
        , m_type(LengthType)
    {
        ASSERT(!length.isUndefined());
    }

    explicit GridLength(double flex)
        : m_flex(flex)
        , m_type(FlexType)
    {
    }

    bool isLength() const { return m_type == LengthType; }
    bool isFlex() const { return m_type == FlexType; }

    const Length& length() const { ASSERT(isLength()); return m_length; }

    double flex() const { ASSERT(isFlex()); return m_flex; }

    bool isPercentage() const { return m_type == LengthType && m_length.isPercentOrCalculated(); }

    friend bool operator==(const GridLength&, const GridLength&) = default;

    bool isContentSized() const { return m_type == LengthType && (m_length.isAuto() || m_length.isMinContent() || m_length.isMaxContent()); }
    bool isAuto() const { return m_type == LengthType && m_length.isAuto(); }

private:
    // Ideally we would put the 2 following fields in a union, but Length has a constructor,
    // a destructor and a copy assignment which isn't allowed.
    Length m_length;
    double m_flex;
    enum GridLengthType {
        LengthType,
        FlexType
    };
    GridLengthType m_type;
};

} // namespace WebCore
