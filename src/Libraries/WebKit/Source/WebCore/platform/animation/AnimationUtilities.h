/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include "IntPoint.h"
#include "IterationCompositeOperation.h"
#include "LayoutPoint.h"

namespace WebCore {

struct BlendingContext {
    double progress { 0 };
    bool isDiscrete { false };
    CompositeOperation compositeOperation { CompositeOperation::Replace };
    IterationCompositeOperation iterationCompositeOperation { IterationCompositeOperation::Replace };
    double currentIteration { 0 };

    BlendingContext(double progress = 0, bool isDiscrete = false, CompositeOperation compositeOperation = CompositeOperation::Replace, IterationCompositeOperation iterationCompositeOperation = IterationCompositeOperation::Replace, double currentIteration = 0)
        : progress(progress)
        , isDiscrete(isDiscrete)
        , compositeOperation(compositeOperation)
        , iterationCompositeOperation(iterationCompositeOperation)
        , currentIteration(currentIteration)
    {
    }

    bool isReplace() const
    {
        return compositeOperation == CompositeOperation::Replace && iterationCompositeOperation == IterationCompositeOperation::Replace;
    }

    void normalizeProgress()
    {
        // https://drafts.csswg.org/web-animations-1/#discrete
        // The property's values cannot be meaningfully combined, thus it is not additive and
        // interpolation swaps from Va to Vb at 50% (p=0.5).
        if (isDiscrete) {
            progress = progress < 0.5 ? 0 : 1;
            compositeOperation = CompositeOperation::Replace;
        }
    }
};

inline int blend(int from, int to, const BlendingContext& context)
{  
    if (context.iterationCompositeOperation == IterationCompositeOperation::Accumulate && context.currentIteration) {
        auto iterationIncrement = static_cast<int>(context.currentIteration * static_cast<double>(to));
        from += iterationIncrement;
        to += iterationIncrement;
    }

    if (context.compositeOperation == CompositeOperation::Replace)
        return static_cast<int>(roundTowardsPositiveInfinity(from + (static_cast<double>(to) - from) * context.progress));
    return static_cast<int>(roundTowardsPositiveInfinity(static_cast<double>(from) + static_cast<double>(from) + static_cast<double>(to - from) * context.progress));
}

inline unsigned blend(unsigned from, unsigned to, const BlendingContext& context)
{
    if (context.iterationCompositeOperation == IterationCompositeOperation::Accumulate && context.currentIteration) {
        auto iterationIncrement = static_cast<unsigned>(context.currentIteration * static_cast<double>(to));
        from += iterationIncrement;
        to += iterationIncrement;
    }

    if (context.compositeOperation == CompositeOperation::Replace)
        return static_cast<unsigned>(lround(from + (static_cast<double>(to) - from) * context.progress));
    return static_cast<unsigned>(lround(from + from + (static_cast<double>(to) - from) * context.progress));
}

inline double blend(double from, double to, const BlendingContext& context)
{  
    if (context.iterationCompositeOperation == IterationCompositeOperation::Accumulate && context.currentIteration) {
        auto iterationIncrement = context.currentIteration * to;
        from += iterationIncrement;
        to += iterationIncrement;
    }

    if (context.compositeOperation == CompositeOperation::Replace)
        return from + (to - from) * context.progress;
    return from + from + (to - from) * context.progress;
}

inline float blend(float from, float to, const BlendingContext& context)
{  
    if (context.iterationCompositeOperation == IterationCompositeOperation::Accumulate && context.currentIteration) {
        auto iterationIncrement = static_cast<float>(context.currentIteration * to);
        from += iterationIncrement;
        to += iterationIncrement;
    }

    if (context.compositeOperation == CompositeOperation::Replace)
        return static_cast<float>(from + (to - from) * context.progress);
    return static_cast<float>(from + from + (to - from) * context.progress);
}

inline LayoutUnit blend(LayoutUnit from, LayoutUnit to, const BlendingContext& context)
{
    return LayoutUnit(blend(from.toFloat(), to.toFloat(), context));
}

inline IntPoint blend(const IntPoint& from, const IntPoint& to, const BlendingContext& context)
{
    return IntPoint(blend(from.x(), to.x(), context),
        blend(from.y(), to.y(), context));
}

inline LayoutPoint blend(const LayoutPoint& from, const LayoutPoint& to, const BlendingContext& context)
{
    return LayoutPoint(blend(from.x(), to.x(), context),
        blend(from.y(), to.y(), context));
}

} // namespace WebCore
