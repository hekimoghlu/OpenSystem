/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioNodeOptions.h"
#include "DistanceModelType.h"
#include "PanningModelType.h"

namespace WebCore {

struct PannerOptions : AudioNodeOptions {
    PanningModelType panningModel { PanningModelType::Equalpower };
    DistanceModelType distanceModel { DistanceModelType::Inverse };
    float positionX { 0 };
    float positionY { 0 };
    float positionZ { 0 };
    float orientationX { 1 };
    float orientationY { 0 };
    float orientationZ { 0 };
    double refDistance { 1 };
    double maxDistance { 10000 };
    double rolloffFactor { 1 };
    double coneInnerAngle { 360 };
    double coneOuterAngle { 360 };
    double coneOuterGain { 0 };
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
