/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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

#ifdef __cplusplus

#include <WebKit/FullscreenTouchSecheuristicParameters.h>

namespace WebKit {

class FullscreenTouchSecheuristic {
public:
    WK_EXPORT double scoreOfNextTouch(CGPoint location);
    WK_EXPORT double scoreOfNextTouch(CGPoint location, const Seconds& deltaTime);
    WK_EXPORT void reset();

    void setParameters(const FullscreenTouchSecheuristicParameters& parameters) { m_parameters = parameters; }
    double requiredScore() const { return m_parameters.requiredScore; }

    void setRampUpSpeed(Seconds speed) { m_parameters.rampUpSpeed = speed; }
    void setRampDownSpeed(Seconds speed) { m_parameters.rampDownSpeed = speed; }
    void setXWeight(double weight) { m_parameters.xWeight = weight; }
    void setYWeight(double weight) { m_parameters.yWeight = weight; }
    void setSize(CGSize size) { m_size = size; }
    void setGamma(double gamma) { m_parameters.gamma = gamma; }
    void setGammaCutoff(double cutoff) { m_parameters.gammaCutoff = cutoff; }

private:
    WK_EXPORT double distanceScore(const CGPoint& nextLocation, const CGPoint& lastLocation, const Seconds& deltaTime);
    WK_EXPORT double attenuationFactor(Seconds delta);

    double m_weight { 0.1 };
    FullscreenTouchSecheuristicParameters m_parameters;
    CGSize m_size { };
    Seconds m_lastTouchTime { 0 };
    CGPoint m_lastTouchLocation { -1, -1 };
    double m_lastScore { 0 };
};

}

#endif
