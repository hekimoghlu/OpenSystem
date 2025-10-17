/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include "SpeechRecognitionRemoteRealtimeMediaSource.h"

#if ENABLE(MEDIA_STREAM)

#include "MessageSenderInlines.h"
#include "SpeechRecognitionRealtimeMediaSourceManagerMessages.h"
#include "SpeechRecognitionRemoteRealtimeMediaSourceManager.h"

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#include <WebCore/CARingBuffer.h>
#include <WebCore/WebAudioBufferList.h>
#endif

namespace WebKit {

Ref<WebCore::RealtimeMediaSource> SpeechRecognitionRemoteRealtimeMediaSource::create(SpeechRecognitionRemoteRealtimeMediaSourceManager& manager, const WebCore::CaptureDevice& captureDevice, WebCore::PageIdentifier pageIdentifier)
{
    return adoptRef(*new SpeechRecognitionRemoteRealtimeMediaSource(WebCore::RealtimeMediaSourceIdentifier::generate(), manager, captureDevice, pageIdentifier));
}

SpeechRecognitionRemoteRealtimeMediaSource::SpeechRecognitionRemoteRealtimeMediaSource(WebCore::RealtimeMediaSourceIdentifier identifier, SpeechRecognitionRemoteRealtimeMediaSourceManager& manager, const WebCore::CaptureDevice& captureDevice, WebCore::PageIdentifier pageIdentifier)
    : WebCore::RealtimeMediaSource(captureDevice, { }, pageIdentifier)
    , m_identifier(identifier)
    , m_manager(manager)
{
    manager.addSource(*this, captureDevice);
}

SpeechRecognitionRemoteRealtimeMediaSource::~SpeechRecognitionRemoteRealtimeMediaSource()
{
    if (RefPtr manager = m_manager.get())
        manager->removeSource(*this);
}

void SpeechRecognitionRemoteRealtimeMediaSource::startProducingData()
{
    if (RefPtr manager = m_manager.get())
        manager->send(Messages::SpeechRecognitionRealtimeMediaSourceManager::Start { m_identifier });
}

void SpeechRecognitionRemoteRealtimeMediaSource::stopProducingData()
{
    if (RefPtr manager = m_manager.get())
        manager->send(Messages::SpeechRecognitionRealtimeMediaSourceManager::Stop { m_identifier });
}

#if PLATFORM(COCOA)

void SpeechRecognitionRemoteRealtimeMediaSource::setStorage(ConsumerSharedCARingBuffer::Handle&& handle, const WebCore::CAAudioStreamDescription& description)
{
    m_buffer = nullptr;
    m_ringBuffer = ConsumerSharedCARingBuffer::map(description, WTFMove(handle));
    if (!m_ringBuffer)
        return;
    m_description = description;
    m_buffer = makeUnique<WebCore::WebAudioBufferList>(description);
}

#endif

void SpeechRecognitionRemoteRealtimeMediaSource::remoteAudioSamplesAvailable(MediaTime time, uint64_t numberOfFrames)
{
#if PLATFORM(COCOA)
    if (!m_buffer) {
        LOG_ERROR("Buffer for remote source is null");
        captureFailed();
        return;
    }

    m_buffer->setSampleCount(numberOfFrames);
    m_ringBuffer->fetch(m_buffer->list(), numberOfFrames, time.timeValue());
    audioSamplesAvailable(time, *m_buffer, *m_description, numberOfFrames);
#else
    UNUSED_PARAM(time);
    UNUSED_PARAM(numberOfFrames);
#endif
}

void SpeechRecognitionRemoteRealtimeMediaSource::remoteCaptureFailed()
{
    captureFailed();
}

void SpeechRecognitionRemoteRealtimeMediaSource::remoteSourceStopped()
{
    stop();
}

} // namespace WebKit

#endif
