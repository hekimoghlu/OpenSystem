/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#include "RemoteMediaRecorderPrivateWriterManager.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "Logging.h"
#include "RemoteMediaRecorderPrivateWriterIdentifier.h"
#include "RemoteMediaResourceManagerMessages.h"
#include <WebCore/MediaSample.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, m_gpuConnectionToWebProcess.get()->connection())
#define MESSAGE_CHECK_COMPLETION(assertion, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, m_gpuConnectionToWebProcess.get()->connection(), completion)

typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaRecorderPrivateWriterManager);

class RemoteMediaRecorderPrivateWriterProxy : public WebCore::MediaRecorderPrivateWriterListener {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaRecorderPrivateWriterProxy);
public:
    static Ref<RemoteMediaRecorderPrivateWriterProxy> create() { return adoptRef(*new RemoteMediaRecorderPrivateWriterProxy()); }

    std::optional<uint8_t> addAudioTrack(const AudioInfo& description)
    {
        return m_writer->addAudioTrack(description);
    }

    std::optional<uint8_t> addVideoTrack(const VideoInfo& description, const std::optional<CGAffineTransform>& transform)
    {
        return m_writer->addVideoTrack(description, transform);
    }

    bool allTracksAdded()
    {
        return m_writer->allTracksAdded();
    }

    Ref<WebCore::MediaRecorderPrivateWriter::WriterPromise> writeFrames(Deque<UniqueRef<MediaSamplesBlock>>&& samples, const MediaTime& time)
    {
        return m_writer->writeFrames(WTFMove(samples), time);
    }

    Ref<GenericPromise> close()
    {
        return m_writer->close();
    }

    Ref<SharedBuffer> takeData()
    {
        Locker locker { m_lock };
        return m_data.takeAsContiguous();
    }

private:
    RemoteMediaRecorderPrivateWriterProxy()
        : m_writer(makeUniqueRefFromNonNullUniquePtr(MediaRecorderPrivateWriter::create(MediaRecorderContainerType::Mp4, *this)))
    {
    }

    void appendData(std::span<const uint8_t> data) final
    {
        Locker locker { m_lock };
        m_data.append(data);
    }

    const UniqueRef<MediaRecorderPrivateWriter> m_writer;
    Lock m_lock;
    SharedBufferBuilder m_data WTF_GUARDED_BY_LOCK(m_lock);
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaRecorderPrivateWriterProxy);

RemoteMediaRecorderPrivateWriterManager::RemoteMediaRecorderPrivateWriterManager(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
    : m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
{
}

RemoteMediaRecorderPrivateWriterManager::~RemoteMediaRecorderPrivateWriterManager() = default;

void RemoteMediaRecorderPrivateWriterManager::ref() const
{
    m_gpuConnectionToWebProcess.get()->ref();
}

void RemoteMediaRecorderPrivateWriterManager::deref() const
{
    m_gpuConnectionToWebProcess.get()->deref();
}

void RemoteMediaRecorderPrivateWriterManager::create(RemoteMediaRecorderPrivateWriterIdentifier identifier)
{
    MESSAGE_CHECK(!m_remoteMediaRecorderPrivateWriters.contains(identifier));

    m_remoteMediaRecorderPrivateWriters.add(identifier, Writer { RemoteMediaRecorderPrivateWriterProxy::create() });
}

void RemoteMediaRecorderPrivateWriterManager::addAudioTrack(RemoteMediaRecorderPrivateWriterIdentifier identifier, RemoteAudioInfo info, CompletionHandler<void(std::optional<uint8_t>)>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(m_remoteMediaRecorderPrivateWriters.contains(identifier), completionHandler(std::nullopt));

    auto iterator = m_remoteMediaRecorderPrivateWriters.find(identifier);
    if (iterator == m_remoteMediaRecorderPrivateWriters.end())
        return;

    Ref audioInfo = info.toAudioInfo();
    iterator->value.audioInfo = audioInfo.ptr();
    RefPtr writer = iterator->value.proxy;
    auto result = writer->addAudioTrack(audioInfo.get());
    if (result)
        audioInfo->trackID = *result;
    completionHandler(result);
}

void RemoteMediaRecorderPrivateWriterManager::addVideoTrack(RemoteMediaRecorderPrivateWriterIdentifier identifier, RemoteVideoInfo info, std::optional<CGAffineTransform> transform, CompletionHandler<void(std::optional<uint8_t>)>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(m_remoteMediaRecorderPrivateWriters.contains(identifier), completionHandler(std::nullopt));

    auto iterator = m_remoteMediaRecorderPrivateWriters.find(identifier);
    if (iterator == m_remoteMediaRecorderPrivateWriters.end())
        return;

    Ref videoInfo = info.toVideoInfo();
    iterator->value.videoInfo = videoInfo.ptr();
    RefPtr writer = iterator->value.proxy;
    auto result = writer->addVideoTrack(videoInfo.get(), transform);
    if (result)
        videoInfo->trackID = *result;
    completionHandler(result);
}

void RemoteMediaRecorderPrivateWriterManager::allTracksAdded(RemoteMediaRecorderPrivateWriterIdentifier identifier, CompletionHandler<void(bool)>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(m_remoteMediaRecorderPrivateWriters.contains(identifier), completionHandler(false));

    RefPtr writer = m_remoteMediaRecorderPrivateWriters.get(identifier).proxy;
    completionHandler(writer ? writer->allTracksAdded() : false);
}

void RemoteMediaRecorderPrivateWriterManager::writeFrames(RemoteMediaRecorderPrivateWriterIdentifier identifier, Vector<BlockPair>&& vectorSamples, const MediaTime& endTime, CompletionHandler<void(Expected<Ref<WebCore::SharedBuffer>, WebCore::MediaRecorderPrivateWriter::Result>)>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(m_remoteMediaRecorderPrivateWriters.contains(identifier), makeUnexpected("Invalid Identifier"));

    auto iterator = m_remoteMediaRecorderPrivateWriters.find(identifier);
    if (iterator == m_remoteMediaRecorderPrivateWriters.end())
        return;

    RefPtr audioInfo = iterator->value.audioInfo;
    RefPtr videoInfo = iterator->value.videoInfo;

    Deque<UniqueRef<MediaSamplesBlock>> samples;
    for (auto& sample : vectorSamples)
        samples.append(makeUniqueRef<MediaSamplesBlock>(sample.first == TrackInfo::TrackType::Audio ? audioInfo.get() : static_cast<TrackInfo*>(videoInfo.get()), WTFMove(sample.second)));

    RefPtr writer = m_remoteMediaRecorderPrivateWriters.get(identifier).proxy;
    writer->writeFrames(WTFMove(samples), endTime)->whenSettled(RunLoop::protectedMain(), [writer, completionHandler = WTFMove(completionHandler)](auto&& result) mutable {
        if (!result) {
            completionHandler(makeUnexpected(result.error()));
            return;
        }
        completionHandler(writer->takeData());
    });
}

void RemoteMediaRecorderPrivateWriterManager::close(RemoteMediaRecorderPrivateWriterIdentifier identifier, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>)>&& completionHandler)
{
    if (!m_remoteMediaRecorderPrivateWriters.contains(identifier)) {
        // Failsafe if already closed.
        completionHandler(SharedBuffer::create());
        return;
    }

    RefPtr writer = m_remoteMediaRecorderPrivateWriters.take(identifier).proxy;
    writer->close()->whenSettled(RunLoop::protectedMain(), [writer, completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(writer->takeData());
    });
}

std::optional<SharedPreferencesForWebProcess> RemoteMediaRecorderPrivateWriterManager::sharedPreferencesForWebProcess() const
{
    if (RefPtr gpuConnectionToWebProcess = m_gpuConnectionToWebProcess.get())
        return gpuConnectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_RECORDER)

