/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

#include "DelayNode.h"

#include "DelayOptions.h"
#include "DelayProcessor.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DelayNode);

constexpr double maximumAllowedDelayTime = 180;

inline DelayNode::DelayNode(BaseAudioContext& context, double maxDelayTime)
    : AudioBasicProcessorNode(context, NodeTypeDelay)
{
    m_processor = makeUnique<DelayProcessor>(context, context.sampleRate(), 1, maxDelayTime);

    // Initialize so that AudioParams can be processed.
    initialize();
}

ExceptionOr<Ref<DelayNode>> DelayNode::create(BaseAudioContext& context, const DelayOptions& options)
{
    if (options.maxDelayTime <= 0)
        return Exception { ExceptionCode::NotSupportedError, "maxDelayTime should be a positive value"_s };

    if (options.maxDelayTime >= maximumAllowedDelayTime || std::isnan(options.maxDelayTime))
        return Exception { ExceptionCode::NotSupportedError, makeString("maxDelayTime should be less than "_s, maximumAllowedDelayTime) };

    auto delayNode = adoptRef(*new DelayNode(context, options.maxDelayTime));

    auto result = delayNode->handleAudioNodeOptions(options, { 2, ChannelCountMode::Max, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    delayNode->delayTime().setValue(options.delayTime);

    return delayNode;
}

AudioParam& DelayNode::delayTime()
{
    return downcast<DelayProcessor>(*m_processor).delayTime();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
