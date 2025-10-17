/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#include "MediaStreamAudioDestinationNode.h"

#if ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)

#include "AudioContext.h"
#include "AudioNodeInput.h"
#include "Document.h"
#include "MediaStream.h"
#include "MediaStreamAudioSource.h"
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaStreamAudioDestinationNode);

ExceptionOr<Ref<MediaStreamAudioDestinationNode>> MediaStreamAudioDestinationNode::create(BaseAudioContext& context, const AudioNodeOptions& options)
{
    // This behavior is not part of the specification. This is done for consistency with Blink.
    if (context.isStopped() || !context.scriptExecutionContext())
        return Exception { ExceptionCode::NotAllowedError, "Cannot create a MediaStreamAudioDestinationNode in a detached frame"_s };

    auto node = adoptRef(*new MediaStreamAudioDestinationNode(context));

    auto result = node->handleAudioNodeOptions(options, { 2, ChannelCountMode::Explicit, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    return node;
}

MediaStreamAudioDestinationNode::MediaStreamAudioDestinationNode(BaseAudioContext& context)
    : AudioBasicInspectorNode(context, NodeTypeMediaStreamAudioDestination)
    , m_source(MediaStreamAudioSource::create(context.sampleRate()))
    , m_stream(MediaStream::create(*context.document(), MediaStreamPrivate::create(context.document()->logger(), m_source.copyRef())))
{
    initialize();
}

MediaStreamAudioDestinationNode::~MediaStreamAudioDestinationNode()
{
    uninitialize();
}

void MediaStreamAudioDestinationNode::process(size_t numberOfFrames)
{
    m_source->consumeAudio(*input(0)->bus(), numberOfFrames);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)
