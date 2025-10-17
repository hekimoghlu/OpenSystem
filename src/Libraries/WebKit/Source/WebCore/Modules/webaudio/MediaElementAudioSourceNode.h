/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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

#if ENABLE(WEB_AUDIO) && ENABLE(VIDEO)

#include "AudioNode.h"
#include "AudioSourceProviderClient.h"
#include "HTMLMediaElement.h"
#include "MultiChannelResampler.h"
#include <memory>
#include <wtf/Lock.h>

namespace WebCore {

class AudioContext;
struct MediaElementAudioSourceOptions;
    
class MediaElementAudioSourceNode final : public AudioNode, public AudioSourceProviderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaElementAudioSourceNode);
public:
    static ExceptionOr<Ref<MediaElementAudioSourceNode>> create(BaseAudioContext&, MediaElementAudioSourceOptions&&);

    virtual ~MediaElementAudioSourceNode();

    USING_CAN_MAKE_WEAKPTR(AudioNode);

    HTMLMediaElement& mediaElement() { return m_mediaElement; }

    // AudioNode
    void process(size_t framesToProcess) override;
    
    // AudioSourceProviderClient
    void setFormat(size_t numberOfChannels, float sampleRate) override;

    Lock& processLock() WTF_RETURNS_LOCK(m_processLock) { return m_processLock; }

private:
    MediaElementAudioSourceNode(BaseAudioContext&, Ref<HTMLMediaElement>&&);
    void provideInput(AudioBus*, size_t framesToProcess);

    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

    // As an audio source, we will never propagate silence.
    bool propagatesSilence() const override { return false; }

    bool wouldTaintOrigin();

    Ref<HTMLMediaElement> m_mediaElement;
    Lock m_processLock;

    unsigned m_sourceNumberOfChannels WTF_GUARDED_BY_LOCK(m_processLock) { 0 };
    double m_sourceSampleRate WTF_GUARDED_BY_LOCK(m_processLock) { 0 };
    bool m_muted WTF_GUARDED_BY_LOCK(m_processLock) { false };

    std::unique_ptr<MultiChannelResampler> m_multiChannelResampler WTF_GUARDED_BY_LOCK(m_processLock);
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && ENABLE(VIDEO)
