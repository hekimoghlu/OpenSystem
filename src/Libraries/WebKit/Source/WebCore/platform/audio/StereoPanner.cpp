/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "StereoPanner.h"

#if ENABLE(WEB_AUDIO)

#include "SharedBuffer.h"
#include "VectorMath.h"
#include <wtf/IndexedRange.h>
#include <wtf/MathExtras.h>

namespace WebCore {

namespace StereoPanner {

void panWithSampleAccurateValues(const AudioBus* inputBus, AudioBus* outputBus, std::span<const float> panValues)
{
    bool isInputSafe = inputBus && (inputBus->numberOfChannels() == 1 || inputBus->numberOfChannels() == 2) && panValues.size() <= inputBus->length();
    ASSERT(isInputSafe);
    if (!isInputSafe)
        return;
    
    unsigned numberOfInputChannels = inputBus->numberOfChannels();
    
    bool isOutputSafe = outputBus && outputBus->numberOfChannels() == 2 && panValues.size() <= outputBus->length();
    ASSERT(isOutputSafe);
    if (!isOutputSafe)
        return;
    
    auto sourceL = inputBus->channel(0)->span();
    auto sourceR = numberOfInputChannels > 1 ? inputBus->channel(1)->span() : sourceL;
    auto destinationL = outputBus->channelByType(AudioBus::ChannelLeft)->mutableSpan();
    auto destinationR = outputBus->channelByType(AudioBus::ChannelRight)->mutableSpan();
    
    double gainL;
    double gainR;
    double panRadian;
    
    // Handles mono source case first, then stereo source case.
    if (numberOfInputChannels == 1) {
        for (auto [i, panValue] : indexedRange(panValues)) {
            float inputL = sourceL[i];
            double pan = clampTo(panValue, -1.0, 1.0);
            // Pan from left to right [-1; 1] will be normalized as [0; 1].
            panRadian = (pan * 0.5 + 0.5) * piOverTwoDouble;
            gainL = cos(panRadian);
            gainR = sin(panRadian);
            destinationL[i] = static_cast<float>(inputL * gainL);
            destinationR[i] = static_cast<float>(inputL * gainR);
        }
    } else {
        for (auto [i, panValue] : indexedRange(panValues)) {
            float inputL = sourceL[i];
            float inputR = sourceR[i];
            double pan = clampTo(panValue, -1.0, 1.0);
            // Normalize [-1; 0] to [0; 1]. Do nothing when [0; 1].
            panRadian = (pan <= 0 ? pan + 1 : pan) * piOverTwoDouble;
            gainL = cos(panRadian);
            gainR = sin(panRadian);
            if (pan <= 0) {
                destinationL[i] = static_cast<float>(inputL + inputR * gainL);
                destinationR[i] = static_cast<float>(inputR * gainR);
            } else {
                destinationL[i] = static_cast<float>(inputL * gainL);
                destinationR[i] = static_cast<float>(inputR + inputL * gainR);
            }
        }
    }
}

void panToTargetValue(const AudioBus* inputBus, AudioBus* outputBus, float panValue, size_t framesToProcess)
{
    bool isInputSafe = inputBus && (inputBus->numberOfChannels() == 1 || inputBus->numberOfChannels() == 2) && framesToProcess <= inputBus->length();
    ASSERT(isInputSafe);
    if (!isInputSafe)
        return;
    
    unsigned numberOfInputChannels = inputBus->numberOfChannels();
    
    bool isOutputSafe = outputBus && outputBus->numberOfChannels() == 2 && framesToProcess <= outputBus->length();
    ASSERT(isOutputSafe);
    if (!isOutputSafe)
        return;
    
    auto sourceL = inputBus->channel(0)->span().first(framesToProcess);
    auto sourceR = numberOfInputChannels > 1 ? inputBus->channel(1)->span().first(framesToProcess) : sourceL;
    auto destinationL = outputBus->channelByType(AudioBus::ChannelLeft)->mutableSpan();
    auto destinationR = outputBus->channelByType(AudioBus::ChannelRight)->mutableSpan();

    float targetPan = clampTo(panValue, -1.0, 1.0);
    
    if (numberOfInputChannels == 1) {
        double panRadian = (targetPan * 0.5 + 0.5) * piOverTwoDouble;
        
        double gainL = cos(panRadian);
        double gainR = sin(panRadian);
        
        VectorMath::multiplyByScalar(sourceL, gainL, destinationL);
        VectorMath::multiplyByScalar(sourceL, gainR, destinationR);
    } else {
        double panRadian = (targetPan <= 0 ? targetPan + 1 : targetPan) * piOverTwoDouble;
        
        double gainL = cos(panRadian);
        double gainR = sin(panRadian);

        if (targetPan <= 0) {
            VectorMath::multiplyByScalarThenAddToVector(sourceR, gainL, sourceL, destinationL);
            VectorMath::multiplyByScalar(sourceR, gainR, destinationR);
        } else {
            VectorMath::multiplyByScalar(sourceL, gainL, destinationL);
            VectorMath::multiplyByScalarThenAddToVector(sourceL, gainR, sourceR, destinationR);
        }
    }
}

} // namespace StereoPanner

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
