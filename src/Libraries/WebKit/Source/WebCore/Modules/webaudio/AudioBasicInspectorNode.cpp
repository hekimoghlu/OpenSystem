/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioBasicInspectorNode.h"

#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioBasicInspectorNode);

AudioBasicInspectorNode::AudioBasicInspectorNode(BaseAudioContext& context, NodeType type)
    : AudioNode(context, type)
{
    addInput();
}

// We override pullInputs() as an optimization allowing this node to take advantage of in-place processing,
// where the input is simply passed through unprocessed to the output.
// Note: this only applies if the input and output channel counts match.
void AudioBasicInspectorNode::pullInputs(size_t framesToProcess)
{
    // Render input stream - try to render directly into output bus for pass-through processing where process() doesn't need to do anything...
    auto* output = this->output(0);
    input(0)->pull(output ? output->bus() : nullptr, framesToProcess);
}

void AudioBasicInspectorNode::checkNumberOfChannelsForInput(AudioNodeInput* input)
{
    ASSERT(context().isAudioThread() && context().isGraphOwner());

    ASSERT(input == this->input(0));
    if (input != this->input(0))
        return;

    if (auto* output = this->output(0)) {
        unsigned numberOfChannels = input->numberOfChannels();

        if (numberOfChannels != output->numberOfChannels()) {
            // This will propagate the channel count to any nodes connected further downstream in the graph.
            output->setNumberOfChannels(numberOfChannels);
        }
    }

    AudioNode::checkNumberOfChannelsForInput(input);

    updatePullStatus();
}

void AudioBasicInspectorNode::updatePullStatus()
{
    ASSERT(context().isGraphOwner());

    auto output = this->output(0);
    if (output && output->isConnected()) {
        // When an AudioBasicInspectorNode is connected to a downstream node, it will get pulled by the
        // downstream node, thus remove it from the context's automatic pull list.
        if (m_needAutomaticPull) {
            context().removeAutomaticPullNode(*this);
            m_needAutomaticPull = false;
        }
    } else {
        unsigned numberOfInputConnections = input(0)->numberOfRenderingConnections();
        if (numberOfInputConnections && !m_needAutomaticPull) {
            // When an AudioBasicInspectorNode is not connected to any downstream node while still connected from
            // upstream node(s), add it to the context's automatic pull list.
            context().addAutomaticPullNode(*this);
            m_needAutomaticPull = true;
        } else if (!numberOfInputConnections && m_needAutomaticPull) {
            // The AudioBasicInspectorNode is connected to nothing and is not an AnalyserNode, remove it from the
            // context's automatic pull list. AnalyserNode's need to be pulled even with no inputs so that the
            // internal state gets updated to hold the right time and FFT data.
            context().removeAutomaticPullNode(*this);
            m_needAutomaticPull = false;
        }
    }
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
