/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include "RenderStyleConstants.h"

namespace WebCore {
    
class LineClampValue {
public:
    constexpr LineClampValue()
        : m_type(LineClamp::LineCount)
        , m_value(-1)
    {
    }
    
    constexpr LineClampValue(int value, LineClamp type)
        : m_type(type)
        , m_value(value)
    {
    }
    
    constexpr int value() const { return m_value; }
    
    constexpr bool isPercentage() const { return m_type == LineClamp::Percentage; }

    constexpr bool isNone() const { return m_value == -1; }

    friend constexpr bool operator==(const LineClampValue&, const LineClampValue&) = default;
    
private:
    LineClamp m_type;
    int m_value;
};
    
} // namespace WebCore
