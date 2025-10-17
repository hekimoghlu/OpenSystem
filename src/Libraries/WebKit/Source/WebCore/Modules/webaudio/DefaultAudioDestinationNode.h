/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

#include "AudioDestinationNode.h"
#include "AudioIOCallback.h"

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioContext;
class AudioDestination;
    
class DefaultAudioDestinationNode final : public AudioDestinationNode, public AudioIOCallback {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DefaultAudioDestinationNode);
public:
    explicit DefaultAudioDestinationNode(AudioContext&, std::optional<float> sampleRate = std::nullopt);
    ~DefaultAudioDestinationNode();

    AudioContext& context();
    const AudioContext& context() const;

    unsigned framesPerBuffer() const;
    WTF::MediaTime outputLatency() const;

    void startRendering(CompletionHandler<void(std::optional<Exception>&&)>&&) final;
    void resume(CompletionHandler<void(std::optional<Exception>&&)>&&);
    void suspend(CompletionHandler<void(std::optional<Exception>&&)>&&);
    void close(CompletionHandler<void()>&&);

    void setMuted(bool muted) { m_muted = muted; }
    bool isPlayingAudio() const { return m_isEffectivelyPlayingAudio; }
    bool isConnected() const;

private:
    void createDestination();
    void clearDestination();
    void recreateDestination();

    // AudioIOCallback
    // The audio hardware calls render() to get the next render quantum of audio into destinationBus.
    // It will optionally give us local/live audio input in sourceBus (if it's not 0).
    void render(AudioBus* sourceBus, AudioBus* destinationBus, size_t numberOfFrames, const AudioIOPosition& outputPosition) final;
    void isPlayingDidChange() final;

    void setIsSilent(bool);
    void updateIsEffectivelyPlayingAudio();

    Function<void(Function<void()>&&)> dispatchToRenderThreadFunction();

    void initialize() final;
    void uninitialize() final;
    ExceptionOr<void> setChannelCount(unsigned) final;

    bool requiresTailProcessing() const final { return false; }

    void enableInput(const String& inputDeviceId) final;
    void restartRendering() final;
    unsigned maxChannelCount() const final;

    RefPtr<AudioDestination> m_destination;
    String m_inputDeviceId;
    unsigned m_numberOfInputChannels { 0 };
    bool m_wasDestinationStarted { false };
    bool m_isEffectivelyPlayingAudio { false };
    bool m_isSilent { true };
    bool m_muted { false };
};

} // namespace WebCore
