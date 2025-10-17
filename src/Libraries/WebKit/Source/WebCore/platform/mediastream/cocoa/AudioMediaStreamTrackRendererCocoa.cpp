/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#include "AudioMediaStreamTrackRendererCocoa.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioMediaStreamTrackRendererUnit.h"
#include "AudioSampleDataSource.h"
#include "CAAudioStreamDescription.h"
#include "LibWebRTCAudioModule.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioMediaStreamTrackRendererCocoa);

AudioMediaStreamTrackRendererCocoa::AudioMediaStreamTrackRendererCocoa(Init&& init)
    : AudioMediaStreamTrackRenderer(WTFMove(init))
    , m_resetObserver([this] { reset(); })
    , m_deviceID(AudioMediaStreamTrackRenderer::defaultDeviceID())
{
}

AudioMediaStreamTrackRendererCocoa::~AudioMediaStreamTrackRendererCocoa()
{
    ASSERT(!m_registeredDataSource);
}

void AudioMediaStreamTrackRendererCocoa::start(CompletionHandler<void()>&& callback)
{
    clear();

    AudioMediaStreamTrackRendererUnit::singleton().retrieveFormatDescription([weakThis = ThreadSafeWeakPtr { *this }, callback = WTFMove(callback)](auto formatDescription) mutable {
        RefPtr protectedThis = weakThis.get();
        if (protectedThis && formatDescription) {
            protectedThis->m_outputDescription = *formatDescription;
            protectedThis->m_shouldRecreateDataSource = true;
        }
        callback();
    });
}

BaseAudioMediaStreamTrackRendererUnit& AudioMediaStreamTrackRendererCocoa::rendererUnit()
{
#if USE(LIBWEBRTC)
    if (RefPtr audioModule = this->audioModule())
        return audioModule->incomingAudioMediaStreamTrackRendererUnit();
#endif
    return AudioMediaStreamTrackRendererUnit::singleton();
}

void AudioMediaStreamTrackRendererCocoa::stop()
{
    ASSERT(isMainThread());

    if (auto source = m_registeredDataSource)
        rendererUnit().removeSource(m_deviceID, *source);
}

void AudioMediaStreamTrackRendererCocoa::clear()
{
    stop();

    setRegisteredDataSource(nullptr);
    m_outputDescription = std::nullopt;
}

void AudioMediaStreamTrackRendererCocoa::setVolume(float volume)
{
    ASSERT(isMainThread());

    AudioMediaStreamTrackRenderer::setVolume(volume);
    if (auto source = m_registeredDataSource)
        source->setVolume(volume);
}

void AudioMediaStreamTrackRendererCocoa::reset()
{
    ASSERT(isMainThread());

    if (auto source = m_registeredDataSource)
        source->recomputeSampleOffset();
}

void AudioMediaStreamTrackRendererCocoa::setAudioOutputDevice(const String& deviceID)
{
    auto registeredDataSource = m_registeredDataSource;

    setRegisteredDataSource(nullptr);

    m_deviceID = deviceID;
    setRegisteredDataSource(WTFMove(registeredDataSource));

    m_shouldRecreateDataSource = true;
}

void AudioMediaStreamTrackRendererCocoa::setRegisteredDataSource(RefPtr<AudioSampleDataSource>&& source)
{
    ASSERT(isMainThread());

    if (m_registeredDataSource)
        rendererUnit().removeSource(m_deviceID, *m_registeredDataSource);

    if (!m_outputDescription)
        return;

    m_registeredDataSource = source;
    if (!m_registeredDataSource)
        return;

    source->setLogger(logger(), logIdentifier());
    source->setVolume(volume());
    rendererUnit().addResetObserver(m_deviceID, m_resetObserver);
    rendererUnit().addSource(m_deviceID, *m_registeredDataSource);
}

static unsigned pollSamplesCount()
{
#if USE(LIBWEBRTC)
    return LibWebRTCAudioModule::PollSamplesCount + 1;
#else
    return 2;
#endif
}

void AudioMediaStreamTrackRendererCocoa::pushSamples(const MediaTime& sampleTime, const PlatformAudioData& audioData, const AudioStreamDescription& description, size_t sampleCount)
{
    ASSERT(!isMainThread());
    ASSERT(description.platformDescription().type == PlatformDescription::CAAudioStreamBasicType);
    RefPtr dataSource = m_dataSource;
    if (!dataSource || m_shouldRecreateDataSource || !dataSource->inputDescription() || *dataSource->inputDescription() != description) {
        DisableMallocRestrictionsForCurrentThreadScope scope;

        // FIXME: For non libwebrtc sources, we can probably reduce poll samples count to 2.
        
        dataSource = AudioSampleDataSource::create(description.sampleRate() * 0.5, *this, pollSamplesCount());

        if (dataSource->setInputFormat(toCAAudioStreamDescription(description))) {
            ERROR_LOG(LOGIDENTIFIER, "Unable to set the input format of data source");
            return;
        }

        if (!m_outputDescription || dataSource->setOutputFormat(*m_outputDescription)) {
            ERROR_LOG(LOGIDENTIFIER, "Unable to set the output format of data source");
            return;
        }

        callOnMainThread([weakThis = ThreadSafeWeakPtr { *this }, newSource = dataSource]() mutable {
            if (RefPtr protectedThis = weakThis.get())
                protectedThis->setRegisteredDataSource(WTFMove(newSource));
        });
        m_dataSource = dataSource;
        m_shouldRecreateDataSource = false;
    }

    dataSource->pushSamples(sampleTime, audioData, sampleCount);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
