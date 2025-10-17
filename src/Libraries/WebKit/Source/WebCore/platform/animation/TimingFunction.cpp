/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#include "config.h"
#include "TimingFunction.h"

#include "SpringSolver.h"
#include "UnitBezier.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, const TimingFunction& timingFunction)
{
    switch (timingFunction.type()) {
    case TimingFunction::Type::LinearFunction: {
        auto& function = uncheckedDowncast<LinearTimingFunction>(timingFunction);
        ts << "linear(";
        for (size_t i = 0; i < function.points().size(); ++i) {
            if (i)
                ts << ", ";

            const auto& point = function.points()[i];
            ts << point.value << ' ' << FormattedCSSNumber::create(point.progress * 100.0) << '%';
        }
        ts << ")";
        break;
    }
    case TimingFunction::Type::CubicBezierFunction: {
        auto& function = uncheckedDowncast<CubicBezierTimingFunction>(timingFunction);
        ts << "cubic-bezier(" << FormattedCSSNumber::create(function.x1()) << ", " << FormattedCSSNumber::create(function.y1()) << ", " <<  FormattedCSSNumber::create(function.x2()) << ", " << FormattedCSSNumber::create(function.y2()) << ")";
        break;
    }
    case TimingFunction::Type::StepsFunction: {
        auto& function = uncheckedDowncast<StepsTimingFunction>(timingFunction);
        ts << "steps(" << function.numberOfSteps();
        if (auto stepPosition = function.stepPosition()) {
            ts << ", ";
            switch (stepPosition.value()) {
            case StepsTimingFunction::StepPosition::JumpStart:
                ts << "jump-start";
                break;

            case StepsTimingFunction::StepPosition::JumpEnd:
                ts << "jump-end";
                break;

            case StepsTimingFunction::StepPosition::JumpNone:
                ts << "jump-none";
                break;

            case StepsTimingFunction::StepPosition::JumpBoth:
                ts << "jump-both";
                break;

            case StepsTimingFunction::StepPosition::Start:
                ts << "start";
                break;

            case StepsTimingFunction::StepPosition::End:
                ts << "end";
                break;
            }
        }
        ts << ")";
        break;
    }
    case TimingFunction::Type::SpringFunction: {
        auto& function = uncheckedDowncast<SpringTimingFunction>(timingFunction);
        ts << "spring(" << FormattedCSSNumber::create(function.mass()) << " " << FormattedCSSNumber::create(function.stiffness()) << " " <<  FormattedCSSNumber::create(function.damping()) << " " << FormattedCSSNumber::create(function.initialVelocity()) << ")";
        break;
    }
    }
    return ts;
}

double TimingFunction::transformProgress(double progress, double duration, Before before) const
{
    switch (type()) {
    case Type::CubicBezierFunction: {
        auto& function = uncheckedDowncast<CubicBezierTimingFunction>(*this);
        if (function.isLinear())
            return progress;
        // The epsilon value we pass to UnitBezier::solve given that the animation is going to run over |dur| seconds. The longer the
        // animation, the more precision we need in the timing function result to avoid ugly discontinuities.
        auto epsilon = 1.0 / (1000.0 * duration);
        return UnitBezier(function.x1(), function.y1(), function.x2(), function.y2()).solve(progress, epsilon);
    }
    case Type::StepsFunction: {
        // https://drafts.csswg.org/css-easing-1/#step-timing-functions
        auto& function = uncheckedDowncast<StepsTimingFunction>(*this);
        auto steps = function.numberOfSteps();
        auto stepPosition = function.stepPosition();
        // 1. Calculate the current step as floor(input progress value Ã— steps).
        auto currentStep = std::floor(progress * steps);
        // 2. If the step position property is start, increment current step by one.
        if (stepPosition == StepsTimingFunction::StepPosition::JumpStart || stepPosition == StepsTimingFunction::StepPosition::Start || stepPosition == StepsTimingFunction::StepPosition::JumpBoth)
            ++currentStep;
        // 3. If both of the following conditions are true:
        //    - the before flag is set, and
        //    - input progress value Ã— steps mod 1 equals zero (that is, if input progress value Ã— steps is integral), then
        //    decrement current step by one.
        if (before == Before::Yes && !fmod(progress * steps, 1))
            currentStep--;
        // 4. If input progress value â‰¥ 0 and current step < 0, let current step be zero.
        if (progress >= 0 && currentStep < 0)
            currentStep = 0;
        // 5. Calculate jumps based on the step position.
        if (stepPosition == StepsTimingFunction::StepPosition::JumpNone)
            --steps;
        else if (stepPosition == StepsTimingFunction::StepPosition::JumpBoth)
            ++steps;
        // 6. If input progress value â‰¤ 1 and current step > jumps, let current step be jumps.
        if (progress <= 1 && currentStep > steps)
            currentStep = steps;
        // 7. The output progress value is current step / jumps.
        return currentStep / steps;
    }
    case Type::SpringFunction: {
        auto& function = uncheckedDowncast<SpringTimingFunction>(*this);
        return SpringSolver(function.mass(), function.stiffness(), function.damping(), function.initialVelocity()).solve(progress * duration);
    }
    case Type::LinearFunction: {
        auto& function = uncheckedDowncast<LinearTimingFunction>(*this);

        auto& points = function.points();
        if (points.size() < 2)
            return progress;

        auto i = points.reverseFindIf([&] (auto& point) {
            return point.progress <= progress;
        });
        if (i == notFound)
            i = 0;
        else if (i == points.size() - 1)
            --i;

        if (points[i].progress == points[i + 1].progress)
            return points[i + 1].value;

        return points[i].value + ((progress - points[i].progress) / (points[i + 1].progress - points[i].progress) * (points[i + 1].value - points[i].value));
    }
    }

    ASSERT_NOT_REACHED();
    return 0;
}

String TimingFunction::cssText() const
{
    if (auto* function = dynamicDowncast<LinearTimingFunction>(*this)) {
        if (function->points().isEmpty())
            return "linear"_s;
    }

    if (auto* function = dynamicDowncast<CubicBezierTimingFunction>(*this)) {
        if (function->x1() == 0.25 && function->y1() == 0.1 && function->x2() == 0.25 && function->y2() == 1.0)
            return "ease"_s;
        if (function->x1() == 0.42 && !function->y1() && function->x2() == 1.0 && function->y2() == 1.0)
            return "ease-in"_s;
        if (!function->x1() && !function->y1() && function->x2() == 0.58 && function->y2() == 1.0)
            return "ease-out"_s;
        if (function->x1() == 0.42 && !function->y1() && function->x2() == 0.58 && function->y2() == 1.0)
            return "ease-in-out"_s;
        return makeString("cubic-bezier("_s, FormattedCSSNumber::create(function->x1()), ", "_s, FormattedCSSNumber::create(function->y1()), ", "_s, FormattedCSSNumber::create(function->x2()), ", "_s, FormattedCSSNumber::create(function->y2()), ')');
    }

    if (auto* function = dynamicDowncast<StepsTimingFunction>(*this)) {
        if (function->stepPosition() == StepsTimingFunction::StepPosition::JumpEnd || function->stepPosition() == StepsTimingFunction::StepPosition::End)
            return makeString("steps("_s, function->numberOfSteps(), ')');
    }

    TextStream stream;
    stream << *this;
    return stream.release();
}

Ref<CubicBezierTimingFunction> CubicBezierTimingFunction::create(TimingFunctionPreset preset)
{
    switch (preset) {
    case TimingFunctionPreset::Ease:
        return create(TimingFunctionPreset::Ease, 0.25, 0.1, 0.25, 1.0);
    case TimingFunctionPreset::EaseIn:
        return create(TimingFunctionPreset::EaseIn, 0.42, 0.0, 1.0, 1.0);
    case TimingFunctionPreset::EaseOut:
        return create(TimingFunctionPreset::EaseOut, 0.0, 0.0, 0.58, 1.0);
    case TimingFunctionPreset::EaseInOut:
        return create(TimingFunctionPreset::EaseInOut, 0.42, 0.0, 0.58, 1.0);
    case TimingFunctionPreset::Custom:
        break;
    }
    ASSERT_NOT_REACHED();
    return create();
}


} // namespace WebCore
