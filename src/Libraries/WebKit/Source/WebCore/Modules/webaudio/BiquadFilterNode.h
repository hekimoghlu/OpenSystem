/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#include "AudioBasicProcessorNode.h"
#include "BiquadFilterOptions.h"
#include "BiquadProcessor.h"

namespace WebCore {

class AudioParam;

class BiquadFilterNode final : public AudioBasicProcessorNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BiquadFilterNode);
public:
    static ExceptionOr<Ref<BiquadFilterNode>> create(BaseAudioContext& context, const BiquadFilterOptions& = { });

    BiquadFilterType type() const;
    void setType(BiquadFilterType);

    AudioParam& frequency() { return biquadProcessor()->parameter1(); }
    AudioParam& q() { return biquadProcessor()->parameter2(); }
    AudioParam& gain() { return biquadProcessor()->parameter3(); }
    AudioParam& detune() { return biquadProcessor()->parameter4(); }

    // Get the magnitude and phase response of the filter at the given
    // set of frequencies (in Hz). The phase response is in radians.
    ExceptionOr<void> getFrequencyResponse(const Ref<Float32Array>& frequencyHz, const Ref<Float32Array>& magResponse, const Ref<Float32Array>& phaseResponse);

private:
    explicit BiquadFilterNode(BaseAudioContext&);

    BiquadProcessor* biquadProcessor() { return downcast<BiquadProcessor>(processor()); }
};

} // namespace WebCore
