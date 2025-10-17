/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include "AudioDSPKernel.h"
#include "IIRFilter.h"
#include "IIRProcessor.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IIRDSPKernel final : public AudioDSPKernel {
    WTF_MAKE_TZONE_ALLOCATED(IIRDSPKernel);
public:
    explicit IIRDSPKernel(IIRProcessor&);

    // Get the magnitude and phase response of the filter at the given set of frequencies (in Hz). The phase response is in radians.
    void getFrequencyResponse(unsigned length, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse);

private:
    IIRProcessor* iirProcessor() { return downcast<IIRProcessor>(processor()); }

    // AudioDSPKernel
    void process(std::span<const float> source, std::span<float> destination) final;
    double tailTime() const final { return m_tailTime; }
    double latencyTime() const final { return 0; }
    void reset() final;
    bool requiresTailProcessing() const final;

    IIRFilter m_iirFilter;
    double m_tailTime;
};

} // namespace WebCore
