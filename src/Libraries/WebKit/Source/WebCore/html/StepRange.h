/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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

#include "Decimal.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class AnyStepHandling : bool { Reject, Default };

enum class RangeLimitations : bool { Valid, Invalid };

class StepRange {
    WTF_MAKE_TZONE_ALLOCATED(StepRange);
public:
    enum StepValueShouldBe {
        StepValueShouldBeReal,
        ParsedStepValueShouldBeInteger,
        ScaledStepValueShouldBeInteger,
    };

    struct StepDescription {
        WTF_MAKE_TZONE_ALLOCATED(StepDescription);
    public:
        int defaultStep { 1 };
        int defaultStepBase { 0 };
        int stepScaleFactor { 1 };
        StepValueShouldBe stepValueShouldBe { StepValueShouldBeReal };

        constexpr StepDescription(int defaultStep, int defaultStepBase, int stepScaleFactor, StepValueShouldBe stepValueShouldBe = StepValueShouldBeReal)
            : defaultStep(defaultStep)
            , defaultStepBase(defaultStepBase)
            , stepScaleFactor(stepScaleFactor)
            , stepValueShouldBe(stepValueShouldBe)
        {
        }

        StepDescription() = default;

        Decimal defaultValue() const
        {
            return defaultStep * stepScaleFactor;
        }
    };

    enum class IsReversible : bool { No, Yes };

    StepRange();
    StepRange(const StepRange&);
    StepRange(const Decimal& stepBase, RangeLimitations, const Decimal& minimum, const Decimal& maximum, const Decimal& step, const StepDescription&, IsReversible = IsReversible::No);
    Decimal acceptableError() const;
    Decimal alignValueForStep(const Decimal& currentValue, const Decimal& newValue) const;
    Decimal clampValue(const Decimal& value) const;
    bool hasStep() const { return m_hasStep; }
    bool hasRangeLimitations() const { return m_hasRangeLimitations; }
    Decimal maximum() const { return m_maximum; }
    Decimal minimum() const { return m_minimum; }
    Decimal stepSnappedMaximum() const;
    static Decimal parseStep(AnyStepHandling, const StepDescription&, StringView);
    Decimal step() const { return m_step; }
    Decimal stepBase() const { return m_stepBase; }
    int stepScaleFactor() const { return m_stepDescription.stepScaleFactor; }
    bool stepMismatch(const Decimal&) const;
    bool isReversible() const { return m_isReversible; }

    // Clamp the middle value according to the step
    Decimal defaultValue() const
    {
        return clampValue((m_minimum + m_maximum) / 2);
    }

    // Map value into 0-1 range
    Decimal proportionFromValue(const Decimal& value) const
    {
        if (m_minimum == m_maximum)
            return 0;

        return (value - m_minimum) / (m_maximum - m_minimum);
    }

    // Map from 0-1 range to value
    Decimal valueFromProportion(const Decimal& proportion) const
    {
        return m_minimum + proportion * (m_maximum - m_minimum);
    }

private:
    StepRange& operator =(const StepRange&);
    Decimal roundByStep(const Decimal& value, const Decimal& base) const;

    const Decimal m_maximum; // maximum must be >= minimum.
    const Decimal m_minimum;
    const Decimal m_step;
    const Decimal m_stepBase;
    const StepDescription m_stepDescription;
    const bool m_hasRangeLimitations { false };
    const bool m_hasStep { false };
    const bool m_isReversible { false };
};

} // namespace WebCore
