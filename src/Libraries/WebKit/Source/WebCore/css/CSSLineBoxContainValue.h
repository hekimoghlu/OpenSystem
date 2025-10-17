/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include "CSSValue.h"
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>

namespace WebCore {

class CSSPrimitiveValue;

enum class LineBoxContain : uint8_t {
    Block           = 1 << 0,
    Inline          = 1 << 1,
    Font            = 1 << 2,
    Glyphs          = 1 << 3,
    Replaced        = 1 << 4,
    InlineBox       = 1 << 5,
    InitialLetter   = 1 << 6,
};

// Used for text-CSSLineBoxContain and box-CSSLineBoxContain
class CSSLineBoxContainValue final : public CSSValue {
public:
    static Ref<CSSLineBoxContainValue> create(OptionSet<LineBoxContain> value)
    {
        return adoptRef(*new CSSLineBoxContainValue(value));
    }

    String customCSSText() const;
    bool equals(const CSSLineBoxContainValue& other) const { return m_value == other.m_value; }
    OptionSet<LineBoxContain> value() const { return m_value; }

private:
    explicit CSSLineBoxContainValue(OptionSet<LineBoxContain>);

    OptionSet<LineBoxContain> m_value;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSLineBoxContainValue, isLineBoxContainValue())
