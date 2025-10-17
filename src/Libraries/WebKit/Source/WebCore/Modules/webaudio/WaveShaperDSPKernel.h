/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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

#include "AudioArray.h"
#include "AudioDSPKernel.h"
#include "DownSampler.h"
#include "UpSampler.h"
#include "WaveShaperProcessor.h"
#include <memory>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// WaveShaperDSPKernel is an AudioDSPKernel and is responsible for non-linear distortion on one channel.

class WaveShaperDSPKernel final : public AudioDSPKernel {
    WTF_MAKE_TZONE_ALLOCATED(WaveShaperDSPKernel);
public:
    explicit WaveShaperDSPKernel(WaveShaperProcessor*);

    // AudioDSPKernel
    void process(std::span<const float> source, std::span<float> destination) final;
    void reset() final;
    double tailTime() const final { return 0; }
    double latencyTime() const final;

    // Oversampling requires more resources, so let's only allocate them if needed.
    void lazyInitializeOversampling();

private:
    // Apply the shaping curve.
    void processCurve(std::span<const float> source, std::span<float> destination);

    // Use up-sampling, process at the higher sample-rate, then down-sample.
    void processCurve2x(std::span<const float> source, std::span<float> destination);
    void processCurve4x(std::span<const float> source, std::span<float> destination);

    bool requiresTailProcessing() const final;

    WaveShaperProcessor* waveShaperProcessor() { return downcast<WaveShaperProcessor>(processor()); }
    const WaveShaperProcessor* waveShaperProcessor() const { return downcast<WaveShaperProcessor>(processor()); }

    // Oversampling.
    std::unique_ptr<AudioFloatArray> m_tempBuffer;
    std::unique_ptr<AudioFloatArray> m_tempBuffer2;
    std::unique_ptr<UpSampler> m_upSampler;
    std::unique_ptr<DownSampler> m_downSampler;
    std::unique_ptr<UpSampler> m_upSampler2;
    std::unique_ptr<DownSampler> m_downSampler2;
};

} // namespace WebCore
