/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef EqualPowerPanner_h
#define EqualPowerPanner_h

#include "Panner.h"

namespace WebCore {

// Common type of stereo panner as found in normal audio mixing equipment.

class EqualPowerPanner final : public Panner {
public:
    EqualPowerPanner();

    void pan(double azimuth, double elevation, const AudioBus* inputBus, AudioBus* outputBuf, size_t framesToProcess) final;
    void panWithSampleAccurateValues(std::span<double> azimuth, std::span<double> elevation, const AudioBus* inputBus, AudioBus* outputBus, size_t framesToProcess) final;

    void reset() override { }

    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

private:
    void calculateDesiredGain(double& desiredGainL, double& desiredGainR, double azimuth, unsigned numberOfChannels);
};

} // namespace WebCore

#endif // EqualPowerPanner_h
