/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#include <memory>

namespace WebCore {

class AudioBus;
class AudioNodeInput;
class AudioProcessor;
    
// AudioBasicProcessorNode is an AudioNode with one input and one output where the input and output have the same number of channels.
class AudioBasicProcessorNode : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioBasicProcessorNode);
public:
    AudioBasicProcessorNode(BaseAudioContext&, NodeType);
    virtual ~AudioBasicProcessorNode();

    // AudioNode
    void process(size_t framesToProcess) override;
    void processOnlyAudioParams(size_t framesToProcess) override;
    void pullInputs(size_t framesToProcess) override;
    void initialize() override;
    void uninitialize() override;

    // Called in the main thread when the number of channels for the input may have changed.
    void checkNumberOfChannelsForInput(AudioNodeInput*) override;

    // Returns the number of channels for both the input and the output.
    unsigned numberOfChannels();

protected:
    double tailTime() const override;
    double latencyTime() const override;
    bool requiresTailProcessing() const override;

    AudioProcessor* processor() { return m_processor.get(); }
    const AudioProcessor* processor() const { return m_processor.get(); }

    float noiseInjectionMultiplier() const override { return 0.01; }

    std::unique_ptr<AudioProcessor> m_processor;
};

} // namespace WebCore
