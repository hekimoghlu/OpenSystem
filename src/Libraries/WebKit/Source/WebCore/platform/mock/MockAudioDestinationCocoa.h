/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

#if ENABLE(WEB_AUDIO) && PLATFORM(COCOA)

#include "AudioDestinationCocoa.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class AudioIOCallback;

class MockAudioDestinationCocoa final : public AudioDestinationCocoa {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MockAudioDestinationCocoa, WEBCORE_EXPORT);
public:
    static Ref<AudioDestination> create(AudioIOCallback& callback, float sampleRate)
    {
        return adoptRef(*new MockAudioDestinationCocoa(callback, sampleRate));
    }

    WEBCORE_EXPORT MockAudioDestinationCocoa(AudioIOCallback&, float sampleRate);
    WEBCORE_EXPORT virtual ~MockAudioDestinationCocoa();

private:
    void startRendering(CompletionHandler<void(bool)>&&) final;
    void stopRendering(CompletionHandler<void(bool)>&&) final;

    void tick();

    Ref<WorkQueue> m_workQueue;
    RunLoop::Timer m_timer;
    size_t m_numberOfFramesToProcess { 384 };
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && PLATFORM(COCOA)
