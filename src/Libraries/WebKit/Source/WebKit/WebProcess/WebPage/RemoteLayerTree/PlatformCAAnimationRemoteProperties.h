/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

#include <WebCore/PlatformCAAnimation.h>
#include <wtf/Forward.h>

namespace WebKit {

struct PlatformCAAnimationRemoteProperties {
    String keyPath;
    WebCore::PlatformCAAnimation::AnimationType animationType { WebCore::PlatformCAAnimation::AnimationType::Basic };

    CFTimeInterval beginTime { 0 };
    double duration { 0 };
    double timeOffset { 0 };
    float repeatCount { 1 };
    float speed { 1 };

    WebCore::PlatformCAAnimation::FillModeType fillMode { WebCore::PlatformCAAnimation::FillModeType::NoFillMode };
    WebCore::PlatformCAAnimation::ValueFunctionType valueFunction { WebCore::PlatformCAAnimation::ValueFunctionType::NoValueFunction };
    RefPtr<WebCore::TimingFunction> timingFunction;

    bool autoReverses { false };
    bool removedOnCompletion { true };
    bool additive { false };
    bool reverseTimingFunctions { false };
    bool hasExplicitBeginTime { false };

    // For basic animations, these vectors have two entries. For keyframe animations, two or more.
    // timingFunctions has n-1 entries.
    using KeyframeValue = std::variant<float, WebCore::Color, WebCore::FloatPoint3D, WebCore::TransformationMatrix, Ref<WebCore::FilterOperation>>;
    Vector<KeyframeValue> keyValues;
    Vector<float> keyTimes;
    Vector<Ref<WebCore::TimingFunction>> timingFunctions;

    Vector<PlatformCAAnimationRemoteProperties> animations;
};

} // namespace WebKit
