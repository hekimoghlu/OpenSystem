/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#include "RectEdges.h"
#include "WritingMode.h"

namespace WebCore {

class LengthBox : public RectEdges<Length> {
public:
    LengthBox()
        : LengthBox(LengthType::Auto)
    {
    }

    explicit LengthBox(LengthType type)
        : RectEdges(Length(type), Length(type), Length(type), Length(type))
    {
    }

    explicit LengthBox(int v)
        : RectEdges(Length(v, LengthType::Fixed), Length(v, LengthType::Fixed), Length(v, LengthType::Fixed), Length(v, LengthType::Fixed))
    {
    }

    LengthBox(int top, int right, int bottom, int left)
        : RectEdges(Length(top, LengthType::Fixed), Length(right, LengthType::Fixed), Length(bottom, LengthType::Fixed), Length(left, LengthType::Fixed))
    {
    }

    LengthBox(Length&& top, Length&& right, Length&& bottom, Length&& left)
        : RectEdges { WTFMove(top), WTFMove(right), WTFMove(bottom), WTFMove(left) }
    {
    }

    LengthBox(const LengthBox&) = default;
    LengthBox& operator=(const LengthBox&) = default;

    bool isZero() const
    {
        return top().isZero() && right().isZero() && bottom().isZero() && left().isZero();
    }
};

using LayoutBoxExtent = RectEdges<LayoutUnit>;
using FloatBoxExtent = RectEdges<float>;
using IntBoxExtent = RectEdges<int>;

using IntOutsets = IntBoxExtent;
using LayoutOptionalOutsets = RectEdges<std::optional<LayoutUnit>>;

inline LayoutBoxExtent toLayoutBoxExtent(const IntBoxExtent& extent)
{
    return { extent.top(), extent.right(), extent.bottom(), extent.left() };
}

inline FloatBoxExtent toFloatBoxExtent(const IntBoxExtent& extent)
{
    return {
        static_cast<float>(extent.top()),
        static_cast<float>(extent.right()),
        static_cast<float>(extent.bottom()),
        static_cast<float>(extent.left()),
    };
}

WTF::TextStream& operator<<(WTF::TextStream&, const LengthBox&);
WTF::TextStream& operator<<(WTF::TextStream&, const IntBoxExtent&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FloatBoxExtent&);

} // namespace WebCore
