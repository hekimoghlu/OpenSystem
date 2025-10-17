/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#include "JSAudioNode.h"

#include "AnalyserNode.h"
#include "AudioBufferSourceNode.h"
#include "AudioDestinationNode.h"
#include "AudioNode.h"
#include "AudioWorkletNode.h"
#include "BiquadFilterNode.h"
#include "ChannelMergerNode.h"
#include "ChannelSplitterNode.h"
#include "ConstantSourceNode.h"
#include "ConvolverNode.h"
#include "DelayNode.h"
#include "DynamicsCompressorNode.h"
#include "GainNode.h"
#include "IIRFilterNode.h"
#include "JSAnalyserNode.h"
#include "JSAudioBufferSourceNode.h"
#include "JSAudioDestinationNode.h"
#include "JSAudioWorkletNode.h"
#include "JSBiquadFilterNode.h"
#include "JSChannelMergerNode.h"
#include "JSChannelSplitterNode.h"
#include "JSConstantSourceNode.h"
#include "JSConvolverNode.h"
#include "JSDelayNode.h"
#include "JSDynamicsCompressorNode.h"
#include "JSGainNode.h"
#include "JSIIRFilterNode.h"
#include "JSMediaElementAudioSourceNode.h"
#include "JSMediaStreamAudioDestinationNode.h"
#include "JSMediaStreamAudioSourceNode.h"
#include "JSOscillatorNode.h"
#include "JSPannerNode.h"
#include "JSScriptProcessorNode.h"
#include "JSStereoPannerNode.h"
#include "JSWaveShaperNode.h"
#include "MediaElementAudioSourceNode.h"
#include "MediaStreamAudioDestinationNode.h"
#include "MediaStreamAudioSourceNode.h"
#include "OscillatorNode.h"
#include "PannerNode.h"
#include "ScriptProcessorNode.h"
#include "StereoPannerNode.h"
#include "WaveShaperNode.h"

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<AudioNode>&& node)
{
    switch (node->nodeType()) {
    case AudioNode::NodeTypeDestination:
        return createWrapper<AudioDestinationNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeOscillator:
        return createWrapper<OscillatorNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeAudioBufferSource:
        return createWrapper<AudioBufferSourceNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeMediaElementAudioSource:
#if ENABLE(VIDEO)
        return createWrapper<MediaElementAudioSourceNode>(globalObject, WTFMove(node));
#else
        return createWrapper<AudioNode>(globalObject, WTFMove(node));
#endif
#if ENABLE(MEDIA_STREAM)
    case AudioNode::NodeTypeMediaStreamAudioDestination:
        return createWrapper<MediaStreamAudioDestinationNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeMediaStreamAudioSource:
        return createWrapper<MediaStreamAudioSourceNode>(globalObject, WTFMove(node));
#else
    case AudioNode::NodeTypeMediaStreamAudioDestination:
    case AudioNode::NodeTypeMediaStreamAudioSource:
        return createWrapper<AudioNode>(globalObject, WTFMove(node));
#endif
    case AudioNode::NodeTypeJavaScript:
        return createWrapper<ScriptProcessorNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeBiquadFilter:
        return createWrapper<BiquadFilterNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypePanner:
        return createWrapper<PannerNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeConvolver:
        return createWrapper<ConvolverNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeDelay:
        return createWrapper<DelayNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeGain:
        return createWrapper<GainNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeChannelSplitter:
        return createWrapper<ChannelSplitterNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeChannelMerger:
        return createWrapper<ChannelMergerNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeAnalyser:
        return createWrapper<AnalyserNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeDynamicsCompressor:
        return createWrapper<DynamicsCompressorNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeWaveShaper:
        return createWrapper<WaveShaperNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeConstant:
        return createWrapper<ConstantSourceNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeStereoPanner:
        return createWrapper<StereoPannerNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeIIRFilter:
        return createWrapper<IIRFilterNode>(globalObject, WTFMove(node));
    case AudioNode::NodeTypeWorklet:
        return createWrapper<AudioWorkletNode>(globalObject, WTFMove(node));
    }
    RELEASE_ASSERT_NOT_REACHED();
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, AudioNode& node)
{
    return wrap(lexicalGlobalObject, globalObject, node);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
