/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

#include "AnalyserNode.h"

#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AnalyserNode);

ExceptionOr<Ref<AnalyserNode>> AnalyserNode::create(BaseAudioContext& context, const AnalyserOptions& options)
{
    auto analyser = adoptRef(*new AnalyserNode(context));
    
    auto result = analyser->handleAudioNodeOptions(options, { 2, ChannelCountMode::Max, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();
    
    result = analyser->setMinMaxDecibels(options.minDecibels, options.maxDecibels);
    if (result.hasException())
        return result.releaseException();
    
    result = analyser->setFftSize(options.fftSize);
    if (result.hasException())
        return result.releaseException();

    result = analyser->setSmoothingTimeConstant(options.smoothingTimeConstant);
    if (result.hasException())
        return result.releaseException();
    
    return analyser;
}

AnalyserNode::AnalyserNode(BaseAudioContext& context)
    : AudioBasicInspectorNode(context, NodeTypeAnalyser)
    , m_analyser { context.noiseInjectionPolicies() }
{
    addOutput(1);
    
    initialize();
}

AnalyserNode::~AnalyserNode()
{
    uninitialize();
}

void AnalyserNode::process(size_t framesToProcess)
{
    AudioBus* outputBus = output(0)->bus();

    if (!isInitialized()) {
        outputBus->zero();
        return;
    }

    AudioBus* inputBus = input(0)->bus();
    
    // Give the analyser the audio which is passing through this AudioNode. This must always
    // be done so that the state of the Analyser reflects the current input.
    m_analyser.writeInput(inputBus, framesToProcess);

    if (!input(0)->isConnected()) {
        outputBus->zero();
        return;
    }

    // For in-place processing, our override of pullInputs() will just pass the audio data through unchanged if the channel count matches from input to output
    // (resulting in inputBus == outputBus). Otherwise, do an up-mix to stereo.
    if (inputBus != outputBus)
        outputBus->copyFrom(*inputBus);
}

ExceptionOr<void> AnalyserNode::setFftSize(unsigned size)
{
    if (!m_analyser.setFftSize(size))
        return Exception { ExceptionCode::IndexSizeError, "fftSize must be power of 2 in the range 32 to 32768."_s };
    return { };
}

ExceptionOr<void> AnalyserNode::setMinMaxDecibels(double minDecibels, double maxDecibels)
{
    if (maxDecibels <= minDecibels)
        return Exception { ExceptionCode::IndexSizeError, "minDecibels must be less than maxDecibels."_s };
    
    m_analyser.setMinDecibels(minDecibels);
    m_analyser.setMaxDecibels(maxDecibels);
    return { };
}

ExceptionOr<void> AnalyserNode::setMinDecibels(double k)
{
    if (k >= maxDecibels())
        return Exception { ExceptionCode::IndexSizeError, "minDecibels must be less than maxDecibels."_s };

    m_analyser.setMinDecibels(k);
    return { };
}

ExceptionOr<void> AnalyserNode::setMaxDecibels(double k)
{
    if (k <= minDecibels())
        return Exception { ExceptionCode::IndexSizeError, "maxDecibels must be greater than minDecibels."_s };

    m_analyser.setMaxDecibels(k);
    return { };
}

ExceptionOr<void> AnalyserNode::setSmoothingTimeConstant(double k)
{
    if (k < 0 || k > 1)
        return Exception { ExceptionCode::IndexSizeError, "Smoothing time constant needs to be between 0 and 1."_s };

    m_analyser.setSmoothingTimeConstant(k);
    return { };
}

bool AnalyserNode::requiresTailProcessing() const
{
    // Tail time is always non-zero so tail processing is required.
    return true;
}

void AnalyserNode::updatePullStatus()
{
    ASSERT(context().isGraphOwner());

    if (output(0)->isConnected()) {
        // When an AudioBasicInspectorNode is connected to a downstream node, it
        // will get pulled by the downstream node, thus remove it from the context's
        // automatic pull list.
        if (m_needAutomaticPull) {
            context().removeAutomaticPullNode(*this);
            m_needAutomaticPull = false;
        }
    } else {
        unsigned numberOfInputConnections = input(0)->numberOfRenderingConnections();
        // When an AnalyserNode is not connected to any downstream node
        // while still connected from upstream node(s), add it to the context's
        // automatic pull list.
        //
        // But don't remove the AnalyserNode if there are no inputs
        // connected to the node. The node needs to be pulled so that the
        // internal state is updated with the correct input signal (of
        // zeroes).
        if (numberOfInputConnections && !m_needAutomaticPull) {
            context().addAutomaticPullNode(*this);
            m_needAutomaticPull = true;
        }
    }
}

bool AnalyserNode::propagatesSilence() const
{
    // An AnalyserNode does actually propogate silence, but to get the
    // time and FFT data updated correctly, process() needs to be
    // called even if all the inputs are silent.
    return false;
}

double AnalyserNode::tailTime() const
{
    return RealtimeAnalyser::MaxFFTSize / static_cast<double>(context().sampleRate());
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
