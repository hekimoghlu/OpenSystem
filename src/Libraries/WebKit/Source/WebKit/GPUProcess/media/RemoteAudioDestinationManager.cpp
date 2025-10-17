/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "RemoteAudioDestinationManager.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)

#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "Logging.h"
#include <WebCore/AudioUtilities.h>
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/ThreadSafeRefCounted.h>

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#include <WebCore/AudioOutputUnitAdaptor.h>
#include <WebCore/CAAudioStreamDescription.h>
#include <WebCore/CARingBuffer.h>
#include <WebCore/WebAudioBufferList.h>
#endif

#define MESSAGE_CHECK(assertion, message) MESSAGE_CHECK_WITH_MESSAGE_BASE(assertion, &connection->connection(), message)
#define MESSAGE_CHECK_COMPLETION(assertion, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, connection->connection(), completion)

namespace WebKit {

class RemoteAudioDestination final
#if PLATFORM(COCOA)
    : public WebCore::AudioUnitRenderer
#endif
{
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RemoteAudioDestination);
public:
    RemoteAudioDestination(GPUConnectionToWebProcess& connection, const String& inputDeviceId, uint32_t numberOfInputChannels, uint32_t numberOfOutputChannels, float sampleRate, float hardwareSampleRate, IPC::Semaphore&& renderSemaphore)
        : m_renderSemaphore(WTFMove(renderSemaphore))
#if !RELEASE_LOG_DISABLED
        , m_logger(connection.logger())
        , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
#endif
#if PLATFORM(COCOA)
        , m_audioOutputUnitAdaptor(*this)
        , m_numOutputChannels(numberOfOutputChannels)
#endif
    {
        ASSERT(isMainRunLoop());
        ALWAYS_LOG(LOGIDENTIFIER);
#if PLATFORM(COCOA)
        m_audioOutputUnitAdaptor.configure(hardwareSampleRate, numberOfOutputChannels);
#endif
    }

    ~RemoteAudioDestination()
    {
        ASSERT(isMainRunLoop());
        ALWAYS_LOG(LOGIDENTIFIER);
        // Make sure we stop audio rendering and wait for it to finish before destruction.
        if (m_isPlaying)
            stop();
    }

#if PLATFORM(COCOA)
    void setSharedMemory(WebCore::SharedMemory::Handle&& handle)
    {
        m_frameCount = WebCore::SharedMemory::map(WTFMove(handle), WebCore::SharedMemory::Protection::ReadWrite);
    }

    void audioSamplesStorageChanged(ConsumerSharedCARingBuffer::Handle&& handle)
    {
        bool wasPlaying = m_isPlaying;
        if (m_isPlaying) {
            stop();
            ASSERT(!m_isPlaying);
            if (m_isPlaying)
                return;
        }
        m_ringBuffer = ConsumerSharedCARingBuffer::map(sizeof(Float32), m_numOutputChannels, WTFMove(handle));
        if (!m_ringBuffer)
            return;
        if (wasPlaying) {
            start();
            ASSERT(m_isPlaying);
        }
    }
#endif

    void start()
    {
#if PLATFORM(COCOA)
        if (m_audioOutputUnitAdaptor.start()) {
            ERROR_LOG(LOGIDENTIFIER, "Failed to start AudioOutputUnit");
            return;
        }

        ALWAYS_LOG(LOGIDENTIFIER);
        m_isPlaying = true;
#endif
    }

    void stop()
    {
#if PLATFORM(COCOA)
        if (m_audioOutputUnitAdaptor.stop()) {
            ERROR_LOG(LOGIDENTIFIER, "Failed to stop AudioOutputUnit");
            return;
        }

        ALWAYS_LOG(LOGIDENTIFIER);
        m_isPlaying = false;
#endif
    }

    bool isPlaying() const { return m_isPlaying; }

    size_t audioUnitLatency() const
    {
#if PLATFORM(COCOA)
        return m_audioOutputUnitAdaptor.outputLatency();
#else
        return 0;
#endif
    }

private:
#if PLATFORM(COCOA)
    void incrementTotalFrameCount(UInt32 numberOfFrames)
    {
        static_assert(std::atomic<UInt32>::is_always_lock_free, "Shared memory atomic usage assumes lock free primitives are used");
        if (m_frameCount)
            WTF::atomicExchangeAdd(spanReinterpretCast<uint32_t>(m_frameCount->mutableSpan()).data(), numberOfFrames);
    }

    OSStatus render(double sampleTime, uint64_t hostTime, UInt32 numberOfFrames, AudioBufferList* ioData)
    {
        ASSERT(!isMainRunLoop());

        OSStatus status = -1;
        if (m_ringBuffer->fetchIfHasEnoughData(ioData, numberOfFrames, m_startFrame)) {
            m_startFrame += numberOfFrames;
            status = noErr;
        }

        incrementTotalFrameCount(numberOfFrames);
        m_renderSemaphore.signal();

        return status;
    }
#endif

    IPC::Semaphore m_renderSemaphore;
    bool m_isPlaying { false };

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "RemoteAudioDestination"_s; }
    WTFLogChannel& logChannel() const { return WebKit2LogMedia; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    Logger& logger() const { return m_logger; }

    Ref<Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

#if PLATFORM(COCOA)
    WebCore::AudioOutputUnitAdaptor m_audioOutputUnitAdaptor;
    RefPtr<WebCore::SharedMemory> m_frameCount;
    const uint32_t m_numOutputChannels;
    std::unique_ptr<ConsumerSharedCARingBuffer> m_ringBuffer;
    uint64_t m_startFrame { 0 };
#endif
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAudioDestinationManager);

RemoteAudioDestinationManager::RemoteAudioDestinationManager(GPUConnectionToWebProcess& connection)
    : m_gpuConnectionToWebProcess(connection)
{
}

RemoteAudioDestinationManager::~RemoteAudioDestinationManager() = default;

void RemoteAudioDestinationManager::ref() const
{
    m_gpuConnectionToWebProcess.get()->ref();
}

void RemoteAudioDestinationManager::deref() const
{
    m_gpuConnectionToWebProcess.get()->deref();
}

void RemoteAudioDestinationManager::createAudioDestination(RemoteAudioDestinationIdentifier identifier, const String& inputDeviceId, uint32_t numberOfInputChannels, uint32_t numberOfOutputChannels, float sampleRate, float hardwareSampleRate, IPC::Semaphore&& renderSemaphore, WebCore::SharedMemory::Handle&& handle, CompletionHandler<void(size_t)>&& completionHandler)
{
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!connection) {
        completionHandler(0);
        return;
    }
    MESSAGE_CHECK(!connection->isLockdownModeEnabled(), "Received a createAudioDestination() message from a webpage in Lockdown mode.");

    auto destination = makeUniqueRef<RemoteAudioDestination>(*connection, inputDeviceId, numberOfInputChannels, numberOfOutputChannels, sampleRate, hardwareSampleRate, WTFMove(renderSemaphore));
#if PLATFORM(COCOA)
    destination->setSharedMemory(WTFMove(handle));
#else
    UNUSED_PARAM(handle);
#endif
    size_t latency = destination->audioUnitLatency();
    m_audioDestinations.add(identifier, WTFMove(destination));
    completionHandler(latency);
}

void RemoteAudioDestinationManager::deleteAudioDestination(RemoteAudioDestinationIdentifier identifier)
{
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return;
    MESSAGE_CHECK(!connection->isLockdownModeEnabled(), "Received a deleteAudioDestination() message from a webpage in Lockdown mode.");

    m_audioDestinations.remove(identifier);

    if (allowsExitUnderMemoryPressure())
        connection->protectedGPUProcess()->tryExitIfUnusedAndUnderMemoryPressure();
}

void RemoteAudioDestinationManager::startAudioDestination(RemoteAudioDestinationIdentifier identifier, CompletionHandler<void(bool, size_t)>&& completionHandler)
{
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return completionHandler(false, 0);
    MESSAGE_CHECK_COMPLETION(!connection->isLockdownModeEnabled(), completionHandler(false, 0));

    bool isPlaying = false;
    size_t latency = 0;
    if (auto* item = m_audioDestinations.get(identifier)) {
        item->start();
        isPlaying = item->isPlaying();
        latency = item->audioUnitLatency();
    }
    completionHandler(isPlaying, latency);
}

void RemoteAudioDestinationManager::stopAudioDestination(RemoteAudioDestinationIdentifier identifier, CompletionHandler<void(bool)>&& completionHandler)
{
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return completionHandler(false);
    MESSAGE_CHECK_COMPLETION(!connection->isLockdownModeEnabled(), completionHandler(false));

    bool isPlaying = false;
    if (auto* item = m_audioDestinations.get(identifier)) {
        item->stop();
        isPlaying = item->isPlaying();
    }
    completionHandler(isPlaying);
}

#if PLATFORM(COCOA)
void RemoteAudioDestinationManager::audioSamplesStorageChanged(RemoteAudioDestinationIdentifier identifier, ConsumerSharedCARingBuffer::Handle&& handle)
{
    if (auto* item = m_audioDestinations.get(identifier))
        item->audioSamplesStorageChanged(WTFMove(handle));
}
#endif

bool RemoteAudioDestinationManager::allowsExitUnderMemoryPressure() const
{
    for (auto& audioDestination : m_audioDestinations.values()) {
        if (audioDestination->isPlaying())
            return false;
    }
    return true;
}

std::optional<SharedPreferencesForWebProcess> RemoteAudioDestinationManager::sharedPreferencesForWebProcess() const
{
    if (RefPtr gpuConnectionToWebProcess = m_gpuConnectionToWebProcess.get())
        return gpuConnectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)
