/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

#include "BiquadFilterNode.h"
#include <JavaScriptCore/Float32Array.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BiquadFilterNode);

ExceptionOr<Ref<BiquadFilterNode>> BiquadFilterNode::create(BaseAudioContext& context, const BiquadFilterOptions& options)
{
    auto node = adoptRef(*new BiquadFilterNode(context));

    auto result = node->handleAudioNodeOptions(options, { 2, ChannelCountMode::Max, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    node->setType(options.type);
    node->q().setValue(options.Q);
    node->detune().setValue(options.detune);
    node->frequency().setValue(options.frequency);
    node->gain().setValue(options.gain);

    return node;
}

BiquadFilterNode::BiquadFilterNode(BaseAudioContext& context)
    : AudioBasicProcessorNode(context, NodeTypeBiquadFilter)
{
    // Initially setup as lowpass filter.
    m_processor = makeUnique<BiquadProcessor>(context, context.sampleRate(), 1, false);

    // Initialize so that AudioParams can be processed.
    initialize();
}

BiquadFilterType BiquadFilterNode::type() const
{
    return const_cast<BiquadFilterNode*>(this)->biquadProcessor()->type();
}

void BiquadFilterNode::setType(BiquadFilterType type)
{
    biquadProcessor()->setType(type);
}

ExceptionOr<void> BiquadFilterNode::getFrequencyResponse(const Ref<Float32Array>& frequencyHz, const Ref<Float32Array>& magResponse, const Ref<Float32Array>& phaseResponse)
{
    unsigned length = frequencyHz->length();
    if (magResponse->length() != length || phaseResponse->length() != length)
        return Exception { ExceptionCode::InvalidAccessError, "The arrays passed as arguments must have the same length"_s };

    if (length)
        biquadProcessor()->getFrequencyResponse(length, frequencyHz->typedSpan(), magResponse->typedMutableSpan(), phaseResponse->typedMutableSpan());
    return { };
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
