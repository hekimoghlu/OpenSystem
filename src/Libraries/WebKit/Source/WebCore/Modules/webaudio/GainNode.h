/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#include "AudioNode.h"
#include "AudioParam.h"
#include "GainOptions.h"
#include <wtf/Threading.h>

namespace WebCore {

class AudioContext;

// GainNode is an AudioNode with one input and one output which applies a gain (volume) change to the audio signal.
// De-zippering (smoothing) is applied when the gain value is changed dynamically.

class GainNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(GainNode);
public:
    static ExceptionOr<Ref<GainNode>> create(BaseAudioContext& context, const GainOptions& = { });

    // AudioNode
    void process(size_t framesToProcess) override;
    void processOnlyAudioParams(size_t framesToProcess) final;

    // Called in the main thread when the number of channels for the input may have changed.
    void checkNumberOfChannelsForInput(AudioNodeInput*) override;

    // JavaScript interface
    AudioParam& gain() { return m_gain.get(); }

private:
    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

    explicit GainNode(BaseAudioContext&);

    AudioFloatArray m_sampleAccurateGainValues;
    Ref<AudioParam> m_gain;
};

} // namespace WebCore
