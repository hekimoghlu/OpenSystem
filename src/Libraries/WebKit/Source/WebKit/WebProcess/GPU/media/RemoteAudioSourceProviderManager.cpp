/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#include "RemoteAudioSourceProviderManager.h"

#include "GPUProcessConnection.h"
#include "Logging.h"
#include "RemoteAudioSourceProvider.h"
#include "RemoteAudioSourceProviderManagerMessages.h"
#include "SharedCARingBuffer.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS)

namespace WebKit {
using namespace WebCore;

RemoteAudioSourceProviderManager::RemoteAudioSourceProviderManager()
    : m_queue(WorkQueue::create("RemoteAudioSourceProviderManager"_s, WorkQueue::QOS::UserInteractive))
{
}

RemoteAudioSourceProviderManager::~RemoteAudioSourceProviderManager()
{
    ASSERT(!m_connection);
}

void RemoteAudioSourceProviderManager::stopListeningForIPC()
{
    setConnection(nullptr);
}

void RemoteAudioSourceProviderManager::setConnection(IPC::Connection* connection)
{
    if (m_connection == connection)
        return;

    if (m_connection)
        m_connection->removeWorkQueueMessageReceiver(Messages::RemoteAudioSourceProviderManager::messageReceiverName());

    m_connection = WTFMove(connection);

    if (m_connection)
        m_connection->addWorkQueueMessageReceiver(Messages::RemoteAudioSourceProviderManager::messageReceiverName(), m_queue, *this);
}

void RemoteAudioSourceProviderManager::addProvider(Ref<RemoteAudioSourceProvider>&& provider)
{
    ASSERT(WTF::isMainRunLoop());
    setConnection(&WebProcess::singleton().ensureGPUProcessConnection().connection());

    m_queue->dispatch([this, protectedThis = Ref { *this }, provider = WTFMove(provider)]() mutable {
        auto identifier = provider->identifier();

        ASSERT(!m_providers.contains(identifier));
        m_providers.add(identifier, makeUnique<RemoteAudio>(WTFMove(provider)));
    });
}

void RemoteAudioSourceProviderManager::removeProvider(MediaPlayerIdentifier identifier)
{
    ASSERT(WTF::isMainRunLoop());

    m_queue->dispatch([this, protectedThis = Ref { *this }, identifier] {
        ASSERT(m_providers.contains(identifier));
        m_providers.remove(identifier);
    });
}

void RemoteAudioSourceProviderManager::audioStorageChanged(MediaPlayerIdentifier identifier, ConsumerSharedCARingBuffer::Handle&& handle, const WebCore::CAAudioStreamDescription& description)
{
    ASSERT(!WTF::isMainRunLoop());

    auto iterator = m_providers.find(identifier);
    if (iterator == m_providers.end()) {
        RELEASE_LOG_ERROR(Media, "Unable to find provider %llu for storageChanged", identifier.toUInt64());
        return;
    }
    iterator->value->setStorage(WTFMove(handle), description);
}

void RemoteAudioSourceProviderManager::audioSamplesAvailable(MediaPlayerIdentifier identifier, uint64_t startFrame, uint64_t numberOfFrames)
{
    ASSERT(!WTF::isMainRunLoop());

    auto iterator = m_providers.find(identifier);
    if (iterator == m_providers.end()) {
        RELEASE_LOG_ERROR(Media, "Unable to find provider %llu for audioSamplesAvailable", identifier.toUInt64());
        return;
    }
    iterator->value->audioSamplesAvailable(startFrame, numberOfFrames);
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAudioSourceProviderManager::RemoteAudio);

RemoteAudioSourceProviderManager::RemoteAudio::RemoteAudio(Ref<RemoteAudioSourceProvider>&& provider)
    : m_provider(WTFMove(provider))
{
}

void RemoteAudioSourceProviderManager::RemoteAudio::setStorage(ConsumerSharedCARingBuffer::Handle&& handle, const WebCore::CAAudioStreamDescription& description)
{
    m_buffer = nullptr;
    handle.takeOwnershipOfMemory(MemoryLedger::Media);
    m_ringBuffer = ConsumerSharedCARingBuffer::map(description, WTFMove(handle));
    if (!m_ringBuffer)
        return;
    m_description = description;
    m_buffer = makeUnique<WebAudioBufferList>(description);
}

void RemoteAudioSourceProviderManager::RemoteAudio::audioSamplesAvailable(uint64_t startFrame, uint64_t numberOfFrames)
{
    if (!m_buffer) {
        RELEASE_LOG_ERROR(Media, "buffer for audio provider %llu is null", m_provider->identifier().toUInt64());
        return;
    }

    if (!WebAudioBufferList::isSupportedDescription(*m_description, numberOfFrames)) {
        RELEASE_LOG_ERROR(Media, "Unable to support description with given number of frames for audio provider %llu", m_provider->identifier().toUInt64());
        return;
    }

    m_buffer->setSampleCount(numberOfFrames);

    m_ringBuffer->fetch(m_buffer->list(), numberOfFrames, startFrame);

    m_provider->audioSamplesAvailable(*m_buffer, *m_description, numberOfFrames);
}

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS)
