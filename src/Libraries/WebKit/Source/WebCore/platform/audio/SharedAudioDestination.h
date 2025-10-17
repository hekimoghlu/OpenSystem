/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

#if ENABLE(WEB_AUDIO)

#include "AudioDestination.h"

namespace WebCore {

class SharedAudioDestinationAdapter;

class SharedAudioDestination final : public AudioDestination, public ThreadSafeRefCounted<SharedAudioDestination, WTF::DestructionThread::Main> {
public:
    using AudioDestinationCreationFunction = Function<Ref<AudioDestination>(AudioIOCallback&)>;
    WEBCORE_EXPORT static Ref<SharedAudioDestination> create(AudioIOCallback&, unsigned numberOfOutputChannels, float sampleRate, AudioDestinationCreationFunction&&);
    WEBCORE_EXPORT virtual ~SharedAudioDestination();

    void ref() const final { return ThreadSafeRefCounted::ref(); }
    void deref() const final { return ThreadSafeRefCounted::deref(); }

    void sharedRender(AudioBus* sourceBus, AudioBus* destinationBus, size_t framesToProcess, const AudioIOPosition&);

private:
    SharedAudioDestination(AudioIOCallback&, unsigned numberOfOutputChannels, float sampleRate, AudioDestinationCreationFunction&&);

    // AudioDestination
    void start(Function<void(Function<void()>&&)>&& dispatchToRenderThread, CompletionHandler<void(bool)>&&) final;
    void stop(CompletionHandler<void(bool)>&&) final;
    bool isPlaying() final { return m_isPlaying; }
    unsigned framesPerBuffer() const final;
    MediaTime outputLatency() const final;

    void setIsPlaying(bool);

    Ref<SharedAudioDestinationAdapter> protectedOutputAdapter() const;

    Lock m_dispatchToRenderThreadLock;
    Function<void(Function<void()>&&)> m_dispatchToRenderThread WTF_GUARDED_BY_LOCK(m_dispatchToRenderThreadLock);

    bool m_isPlaying { false };
    Ref<SharedAudioDestinationAdapter> m_outputAdapter;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
