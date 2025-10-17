/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#include "StyleStepsEasingFunction.h"

#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "TimingFunction.h"

namespace WebCore {
namespace Style {

CSS::StepsEasingFunction toCSSStepsEasingFunction(const StepsTimingFunction& function, const RenderStyle&)
{
    auto position = function.stepPosition();
    if (!position)
        return { { CSS::StepsEasingParameters::JumpEnd { function.numberOfSteps() } } };

    switch (*position) {
    case StepsTimingFunction::StepPosition::JumpStart:
        return { { CSS::StepsEasingParameters::JumpStart { function.numberOfSteps() } } };

    case StepsTimingFunction::StepPosition::JumpEnd:
        return { { CSS::StepsEasingParameters::JumpEnd { function.numberOfSteps() } } };

    case StepsTimingFunction::StepPosition::JumpNone:
        return { { CSS::StepsEasingParameters::JumpNone { function.numberOfSteps() } } };

    case StepsTimingFunction::StepPosition::JumpBoth:
        return { { CSS::StepsEasingParameters::JumpBoth { function.numberOfSteps() } } };

    case StepsTimingFunction::StepPosition::Start:
        return { { CSS::StepsEasingParameters::Start { function.numberOfSteps() } } };

    case StepsTimingFunction::StepPosition::End:
        return { { CSS::StepsEasingParameters::End { function.numberOfSteps() } } };
    }

    RELEASE_ASSERT_NOT_REACHED();
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::JumpStart)
{
    return StepsTimingFunction::StepPosition::JumpStart;
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::JumpEnd)
{
    return StepsTimingFunction::StepPosition::JumpEnd;
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::JumpBoth)
{
    return StepsTimingFunction::StepPosition::JumpBoth;
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::Start)
{
    return StepsTimingFunction::StepPosition::Start;
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::End)
{
    return StepsTimingFunction::StepPosition::End;
}

static StepsTimingFunction::StepPosition toStepPosition(CSS::Keyword::JumpNone)
{
    return StepsTimingFunction::StepPosition::JumpNone;
}

Ref<TimingFunction> createTimingFunction(const CSS::StepsEasingFunction& function, const CSSToLengthConversionData& conversionData)
{
    return WTF::switchOn(function->value,
        [&](const auto& value) -> Ref<TimingFunction> {
            return StepsTimingFunction::create(toStyle(value.steps, conversionData).value, toStepPosition(value.keyword));
        }
    );
}

Ref<TimingFunction> createTimingFunctionDeprecated(const CSS::StepsEasingFunction& function)
{
    if (!CSS::collectComputedStyleDependencies(function).canResolveDependenciesWithConversionData({ }))
        return StepsTimingFunction::create();

    return WTF::switchOn(function->value,
        [&](const auto& value) -> Ref<TimingFunction> {
            return StepsTimingFunction::create(toStyleNoConversionDataRequired(value.steps).value, toStepPosition(value.keyword));
        }
    );
}

} // namespace Style
} // namespace WebCore
