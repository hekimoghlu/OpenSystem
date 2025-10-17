/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#include "SpeechRecognitionCaptureSourceImpl.h"

#if ENABLE(MEDIA_STREAM)

#include "SpeechRecognitionUpdate.h"
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "CAAudioStreamDescription.h"
#include "WebAudioBufferList.h"
#endif

namespace WebCore {

#if !RELEASE_LOG_DISABLED
static uint64_t nextLogIdentifier()
{
    static uint64_t logIdentifier = cryptographicallyRandomNumber<uint32_t>();
    return ++logIdentifier;
}

static RefPtr<Logger>& nullLogger()
{
    static NeverDestroyed<RefPtr<Logger>> logger;
    return logger;
}
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechRecognitionCaptureSourceImpl);

SpeechRecognitionCaptureSourceImpl::SpeechRecognitionCaptureSourceImpl(SpeechRecognitionConnectionClientIdentifier identifier, DataCallback&& dataCallback, StateUpdateCallback&& stateUpdateCallback, Ref<RealtimeMediaSource>&& source)
    : m_clientIdentifier(identifier)
    , m_dataCallback(WTFMove(dataCallback))
    , m_stateUpdateCallback(WTFMove(stateUpdateCallback))
    , m_source(WTFMove(source))
{
#if !RELEASE_LOG_DISABLED
    if (!nullLogger().get()) {
        nullLogger() = Logger::create(this);
        nullLogger()->setEnabled(this, false);
    }

    m_source->setLogger(*nullLogger(), nextLogIdentifier());
#endif

    m_source->addAudioSampleObserver(*this);
    m_source->addObserver(*this);
    m_source->start();

    initializeWeakPtrFactory();
}

SpeechRecognitionCaptureSourceImpl::~SpeechRecognitionCaptureSourceImpl()
{
    m_source->removeAudioSampleObserver(*this);
    m_source->removeObserver(*this);
    m_source->stop();
}

#if PLATFORM(COCOA)
void SpeechRecognitionCaptureSourceImpl::pullSamplesAndCallDataCallback(AudioSampleDataSource* dataSource, const MediaTime& time, const CAAudioStreamDescription& audioDescription, size_t sampleCount)
{
    ASSERT(isMainThread());

    auto data = WebAudioBufferList { audioDescription, sampleCount };
    {
        Locker locker { m_dataSourceLock };
        if (m_dataSource.get() != dataSource)
            return;
        
        m_dataSource->pullSamples(*data.list(), sampleCount, time.timeValue(), 0, AudioSampleDataSource::Copy);
    }

    m_dataCallback(time, data, audioDescription, sampleCount);
}
#endif

void SpeechRecognitionCaptureSourceImpl::audioSamplesAvailable(const WTF::MediaTime& time, const PlatformAudioData& data, const AudioStreamDescription& description, size_t sampleCount)
{
    if (isMainThread())
        return m_dataCallback(time, data, description, sampleCount);

#if PLATFORM(COCOA)
    // Heap allocations are forbidden on the audio thread for performance reasons so we need to
    // explicitly allow the following allocation(s).
    DisableMallocRestrictionsForCurrentThreadScope scope;
    ASSERT(description.platformDescription().type == PlatformDescription::CAAudioStreamBasicType);
    // Use tryLock() to avoid contention in the real-time audio thread.
    if (!m_dataSourceLock.tryLock())
        return;

    Locker locker { AdoptLock, m_dataSourceLock };
    auto audioDescription = toCAAudioStreamDescription(description);
    if (!m_dataSource || !m_dataSource->inputDescription() || *m_dataSource->inputDescription() != description) {
        auto dataSource = AudioSampleDataSource::create(audioDescription.sampleRate(), m_source.get());
        if (dataSource->setInputFormat(audioDescription)) {
            callOnMainThread([this, weakThis = WeakPtr { *this }] {
                if (!weakThis)
                    return;
    
                m_stateUpdateCallback(SpeechRecognitionUpdate::createError(m_clientIdentifier, SpeechRecognitionError { SpeechRecognitionErrorType::AudioCapture, "Unable to set input format"_s }));
            });
            return;
        }
        
        if (dataSource->setOutputFormat(audioDescription)) {
            callOnMainThread([this, weakThis = WeakPtr { *this }] {
                if (!weakThis)
                    return;

                m_stateUpdateCallback(SpeechRecognitionUpdate::createError(m_clientIdentifier, SpeechRecognitionError { SpeechRecognitionErrorType::AudioCapture, "Unable to set output format"_s }));
            });
            return;
        }
        m_dataSource = WTFMove(dataSource);
    }

    m_dataSource->pushSamples(time, data, sampleCount);
    callOnMainThread([this, weakThis = WeakPtr { *this }, dataSource = m_dataSource, time, audioDescription, sampleCount] {
        if (!weakThis)
            return;

        pullSamplesAndCallDataCallback(dataSource.get(), time, audioDescription, sampleCount);
    });
#endif
}

void SpeechRecognitionCaptureSourceImpl::sourceStarted()
{
    ASSERT(isMainThread());
    m_stateUpdateCallback(SpeechRecognitionUpdate::create(m_clientIdentifier, SpeechRecognitionUpdateType::AudioStart));
}

void SpeechRecognitionCaptureSourceImpl::sourceStopped()
{
    ASSERT(isMainThread());
    ASSERT(m_source->captureDidFail());
    m_stateUpdateCallback(SpeechRecognitionUpdate::createError(m_clientIdentifier, SpeechRecognitionError { SpeechRecognitionErrorType::AudioCapture, "Source is stopped"_s }));
}

void SpeechRecognitionCaptureSourceImpl::sourceMutedChanged()
{
    ASSERT(isMainThread());
    m_stateUpdateCallback(SpeechRecognitionUpdate::createError(m_clientIdentifier, SpeechRecognitionError { SpeechRecognitionErrorType::AudioCapture, "Source is muted"_s }));
}

void SpeechRecognitionCaptureSourceImpl::mute()
{
    m_source->setMuted(true);
}

} // namespace WebCore

#endif
