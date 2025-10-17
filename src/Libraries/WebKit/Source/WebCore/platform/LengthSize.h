/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

#include "CompositeOperation.h"
#include "Length.h"

namespace WebCore {

struct BlendingContext;

struct LengthSize {
    Length width;
    Length height;

    ALWAYS_INLINE friend bool operator==(const LengthSize&, const LengthSize&) = default;

    bool isEmpty() const { return width.isZero() || height.isZero(); }
    bool isZero() const { return width.isZero() && height.isZero(); }
};

inline LengthSize blend(const LengthSize& from, const LengthSize& to, const BlendingContext& context)
{
    return { blend(from.width, to.width, context), blend(from.height, to.height, context) };
}

inline LengthSize blend(const LengthSize& from, const LengthSize& to, const BlendingContext& context, ValueRange valueRange)
{
    return { blend(from.width, to.width, context, valueRange), blend(from.height, to.height, context, valueRange) };
}

WTF::TextStream& operator<<(WTF::TextStream&, const LengthSize&);

} // namespace WebCore
