/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include "AnimationEffectPhase.h"
#include "BasicEffectTiming.h"
#include "FillMode.h"
#include "PlaybackDirection.h"
#include "TimingFunction.h"
#include "WebAnimationTypes.h"
#include <wtf/RefPtr.h>
#include <wtf/Seconds.h>

namespace WebCore {

class WebAnimation;

struct ResolvedEffectTiming {
    MarkableDouble currentIteration;
    AnimationEffectPhase phase { AnimationEffectPhase::Idle };
    MarkableDouble transformedProgress;
    MarkableDouble simpleIterationProgress;
    TimingFunction::Before before;
};

struct AnimationEffectTiming {
    RefPtr<TimingFunction> timingFunction { LinearTimingFunction::create() };
    FillMode fill { FillMode::Auto };
    PlaybackDirection direction { PlaybackDirection::Normal };
    double iterationStart { 0 };
    double iterations { 1 };
    Seconds specifiedStartDelay { 0_s };
    Seconds specifiedEndDelay { 0_s };
    std::optional<Seconds> specifiedIterationDuration;
    WebAnimationTime startDelay { 0_s };
    WebAnimationTime endDelay { 0_s };
    WebAnimationTime iterationDuration { 0_s };
    WebAnimationTime intrinsicIterationDuration { 0_s };
    WebAnimationTime activeDuration { 0_s };
    WebAnimationTime endTime { 0_s };

    struct ResolutionData {
        std::optional<WebAnimationTime> timelineTime;
        std::optional<WebAnimationTime> timelineDuration;
        std::optional<WebAnimationTime> startTime;
        std::optional<WebAnimationTime> localTime;
        double playbackRate { 0 };
    };

    void updateComputedProperties(std::optional<WebAnimationTime> timelineDuration, double playbackRate);
    BasicEffectTiming getBasicTiming(const ResolutionData&) const;
    ResolvedEffectTiming resolve(const ResolutionData&) const;
};

} // namespace WebCore
