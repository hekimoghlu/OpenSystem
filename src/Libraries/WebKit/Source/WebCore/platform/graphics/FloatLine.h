/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#include "FloatPoint.h"
#include "FloatQuad.h"
#include "FloatRect.h"
#include <optional>

namespace WebCore {

class FloatLine {
public:
    FloatLine() = default;
    FloatLine(const FloatPoint& start, const FloatPoint& end)
        : m_start(start)
        , m_end(end)
        , m_length(sqrtf(powf(start.x() - end.x(), 2) + powf(start.y() - end.y(), 2)))
    {
    }
    
    const FloatPoint& start() const { return m_start; }
    const FloatPoint& end() const { return m_end; }
    
    float length() const { return m_length; }

    WEBCORE_EXPORT const FloatPoint pointAtAbsoluteDistance(float) const;
    const FloatPoint pointAtRelativeDistance(float) const;
    const FloatLine extendedToBounds(const FloatRect&) const;
    const std::optional<FloatPoint> intersectionWith(const FloatLine&) const;
    
private:
    FloatPoint m_start { 0, 0 };
    FloatPoint m_end { 0, 0 };
    float m_length { 0 };
};

}
