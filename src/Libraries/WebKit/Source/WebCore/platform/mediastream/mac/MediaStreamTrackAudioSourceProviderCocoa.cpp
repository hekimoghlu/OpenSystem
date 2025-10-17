/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#import "config.h"
#import "MediaStreamTrackAudioSourceProviderCocoa.h"

#if ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)

#import "LibWebRTCAudioModule.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaStreamTrackAudioSourceProviderCocoa);

Ref<MediaStreamTrackAudioSourceProviderCocoa> MediaStreamTrackAudioSourceProviderCocoa::create(MediaStreamTrackPrivate& source)
{
    return adoptRef(*new MediaStreamTrackAudioSourceProviderCocoa(source));
}

MediaStreamTrackAudioSourceProviderCocoa::MediaStreamTrackAudioSourceProviderCocoa(MediaStreamTrackPrivate& source)
    : m_captureSource(source)
    , m_source(source.source())
{
#if USE(LIBWEBRTC)
    if (m_source->isIncomingAudioSource())
        setPollSamplesCount(LibWebRTCAudioModule::PollSamplesCount + 1);
#endif
}

MediaStreamTrackAudioSourceProviderCocoa::~MediaStreamTrackAudioSourceProviderCocoa()
{
    ASSERT(!m_connected);
    m_source->removeAudioSampleObserver(*this);
}

void MediaStreamTrackAudioSourceProviderCocoa::hasNewClient(AudioSourceProviderClient* client)
{
    bool shouldBeConnected = !!client;
    if (m_connected == shouldBeConnected)
        return;

    m_connected = shouldBeConnected;
    if (!client) {
        if (m_captureSource)
            m_captureSource->removeObserver(*this);
        m_source->removeAudioSampleObserver(*this);
        return;
    }

    m_enabled = m_captureSource->enabled();
    m_captureSource->addObserver(*this);
    m_source->addAudioSampleObserver(*this);
}

void MediaStreamTrackAudioSourceProviderCocoa::trackEnabledChanged(MediaStreamTrackPrivate& track)
{
    m_enabled = track.enabled();
}

// May get called on a background thread.
void MediaStreamTrackAudioSourceProviderCocoa::audioSamplesAvailable(const WTF::MediaTime&, const PlatformAudioData& data, const AudioStreamDescription& description, size_t frameCount)
{
    if (!m_enabled)
        return;

    receivedNewAudioSamples(data, description, frameCount);
}

}

#endif // ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)
