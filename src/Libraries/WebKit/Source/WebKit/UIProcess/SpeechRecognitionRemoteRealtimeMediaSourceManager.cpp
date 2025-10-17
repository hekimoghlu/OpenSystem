/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include "SpeechRecognitionRemoteRealtimeMediaSourceManager.h"
#include "MessageSenderInlines.h"

#if ENABLE(MEDIA_STREAM)

#include "SpeechRecognitionRealtimeMediaSourceManagerMessages.h"
#include "SpeechRecognitionRemoteRealtimeMediaSource.h"
#include "WebProcessProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechRecognitionRemoteRealtimeMediaSourceManager);

SpeechRecognitionRemoteRealtimeMediaSourceManager::SpeechRecognitionRemoteRealtimeMediaSourceManager(const WebProcessProxy& process)
    : m_process(process)
{
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::ref() const
{
    m_process->ref();
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::deref() const
{
    m_process->deref();
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::addSource(SpeechRecognitionRemoteRealtimeMediaSource& source, const WebCore::CaptureDevice& captureDevice)
{
    auto identifier = source.identifier();
    ASSERT(!m_sources.contains(identifier));
    m_sources.add(identifier, source);

    send(Messages::SpeechRecognitionRealtimeMediaSourceManager::CreateSource(identifier, captureDevice, *source.pageIdentifier()));
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::removeSource(SpeechRecognitionRemoteRealtimeMediaSource& source)
{
    auto identifier = source.identifier();
    ASSERT(!m_sources.get(identifier).get().get() || m_sources.get(identifier).get().get() == &source);
    m_sources.remove(identifier);

    send(Messages::SpeechRecognitionRealtimeMediaSourceManager::DeleteSource(identifier));
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::remoteAudioSamplesAvailable(WebCore::RealtimeMediaSourceIdentifier identifier, const WTF::MediaTime& time, uint64_t numberOfFrames)
{
    if (auto source = m_sources.get(identifier).get())
        source->remoteAudioSamplesAvailable(time, numberOfFrames);
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::remoteCaptureFailed(WebCore::RealtimeMediaSourceIdentifier identifier)
{
    if (auto source = m_sources.get(identifier).get())
        source->remoteCaptureFailed();
}

void SpeechRecognitionRemoteRealtimeMediaSourceManager::remoteSourceStopped(WebCore::RealtimeMediaSourceIdentifier identifier)
{
    if (auto source = m_sources.get(identifier).get())
        source->remoteSourceStopped();
}

IPC::Connection* SpeechRecognitionRemoteRealtimeMediaSourceManager::messageSenderConnection() const
{
    return &m_process->connection();
}

uint64_t SpeechRecognitionRemoteRealtimeMediaSourceManager::messageSenderDestinationID() const
{
    return 0;
}

#if PLATFORM(COCOA)

void SpeechRecognitionRemoteRealtimeMediaSourceManager::setStorage(WebCore::RealtimeMediaSourceIdentifier identifier, ConsumerSharedCARingBuffer::Handle&& handle, const WebCore::CAAudioStreamDescription& description)
{
    if (auto source = m_sources.get(identifier).get())
        source->setStorage(WTFMove(handle), description);
}

#endif

std::optional<SharedPreferencesForWebProcess> SpeechRecognitionRemoteRealtimeMediaSourceManager::sharedPreferencesForWebProcess() const
{
    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_process->sharedPreferencesForWebProcess();
}

} // namespace WebKit

#endif
