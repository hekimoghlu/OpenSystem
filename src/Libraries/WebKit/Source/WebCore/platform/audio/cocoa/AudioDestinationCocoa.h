/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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

#include "AudioDestinationResampler.h"
#include "AudioOutputUnitAdaptor.h"
#include <AudioUnit/AudioUnit.h>
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class AudioBus;
class MultiChannelResampler;
class PushPullFIFO;

using CreateAudioDestinationCocoaOverride = Ref<AudioDestination>(*)(AudioIOCallback&, float sampleRate);

// An AudioDestination using CoreAudio's default output AudioUnit
class AudioDestinationCocoa : public AudioDestinationResampler, public AudioUnitRenderer, public ThreadSafeRefCounted<AudioDestinationCocoa, WTF::DestructionThread::Main> {
public:
    WEBCORE_EXPORT AudioDestinationCocoa(AudioIOCallback&, unsigned numberOfOutputChannels, float sampleRate);
    WEBCORE_EXPORT virtual ~AudioDestinationCocoa();
    void ref() const final { return ThreadSafeRefCounted<AudioDestinationCocoa, WTF::DestructionThread::Main>::ref(); }
    void deref() const final { return ThreadSafeRefCounted<AudioDestinationCocoa, WTF::DestructionThread::Main>::deref(); }

    WEBCORE_EXPORT static CreateAudioDestinationCocoaOverride createOverride;

protected:
    WEBCORE_EXPORT OSStatus render(double sampleTime, uint64_t hostTime, UInt32 numberOfFrames, AudioBufferList* ioData) final;

private:
    friend Ref<AudioDestination> AudioDestination::create(AudioIOCallback&, const String&, unsigned, unsigned, float);

    void startRendering(CompletionHandler<void(bool)>&&) override;
    void stopRendering(CompletionHandler<void(bool)>&&) override;
    MediaTime outputLatency() const final;

    AudioOutputUnitAdaptor m_audioOutputUnitAdaptor;

};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
