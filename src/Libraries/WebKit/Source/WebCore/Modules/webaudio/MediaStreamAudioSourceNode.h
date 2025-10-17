/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#if ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)

#include "AudioNode.h"
#include "AudioSourceProviderClient.h"
#include "MediaStream.h"
#include "MultiChannelResampler.h"
#include <wtf/Lock.h>

namespace WebCore {

class AudioContext;
struct MediaStreamAudioSourceOptions;
class MultiChannelResampler;
class WebAudioSourceProvider;

class MediaStreamAudioSourceNode final : public AudioNode, public AudioSourceProviderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaStreamAudioSourceNode);
public:
    static ExceptionOr<Ref<MediaStreamAudioSourceNode>> create(BaseAudioContext&, MediaStreamAudioSourceOptions&&);

    ~MediaStreamAudioSourceNode();

    MediaStream& mediaStream() { return m_mediaStream; }

private:
    MediaStreamAudioSourceNode(BaseAudioContext&, MediaStream&, Ref<WebAudioSourceProvider>&&);

    // AudioNode
    void process(size_t framesToProcess) final;
    // AudioSourceProviderClient
    void setFormat(size_t numberOfChannels, float sampleRate) final;

    void provideInput(AudioBus*, size_t framesToProcess);

    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

    // As an audio source, we will never propagate silence.
    bool propagatesSilence() const override { return false; }

    Ref<MediaStream> m_mediaStream;
    Ref<WebAudioSourceProvider> m_provider;
    std::unique_ptr<MultiChannelResampler> m_multiChannelResampler WTF_GUARDED_BY_LOCK(m_processLock);

    Lock m_processLock;

    unsigned m_sourceNumberOfChannels WTF_GUARDED_BY_LOCK(m_processLock) { 0 };
    double m_sourceSampleRate WTF_GUARDED_BY_LOCK(m_processLock) { 0 };
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)
