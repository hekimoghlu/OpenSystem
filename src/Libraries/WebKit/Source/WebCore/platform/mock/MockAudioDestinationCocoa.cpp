/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#include "MockAudioDestinationCocoa.h"

#if ENABLE(WEB_AUDIO)

#include "AudioUtilitiesCocoa.h"
#include "CAAudioStreamDescription.h"
#include "WebAudioBufferList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MockAudioDestinationCocoa);

const int kRenderBufferSize = 128;

MockAudioDestinationCocoa::MockAudioDestinationCocoa(AudioIOCallback& callback, float sampleRate)
    : AudioDestinationCocoa(callback, 2, sampleRate)
    , m_workQueue(WorkQueue::create("MockAudioDestinationCocoa Render Queue"_s))
    , m_timer(RunLoop::current(), this, &MockAudioDestinationCocoa::tick)
{
}

MockAudioDestinationCocoa::~MockAudioDestinationCocoa() = default;

void MockAudioDestinationCocoa::startRendering(CompletionHandler<void(bool)>&& completionHandler)
{
    m_timer.startRepeating(Seconds { m_numberOfFramesToProcess / sampleRate() });
    setIsPlaying(true);

    callOnMainThread([completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(true);
    });
}

void MockAudioDestinationCocoa::stopRendering(CompletionHandler<void(bool)>&& completionHandler)
{
    m_timer.stop();
    setIsPlaying(false);

    callOnMainThread([completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(true);
    });
}

void MockAudioDestinationCocoa::tick()
{
    m_workQueue->dispatch([this, protectedThis = Ref { *this }, sampleRate = sampleRate(), numberOfFramesToProcess = m_numberOfFramesToProcess] {
        AudioStreamBasicDescription streamFormat = audioStreamBasicDescriptionForAudioBus(m_outputBus);
        WebAudioBufferList webAudioBufferList { streamFormat, numberOfFramesToProcess };
        render(0., 0, numberOfFramesToProcess, webAudioBufferList.list());
    });
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
