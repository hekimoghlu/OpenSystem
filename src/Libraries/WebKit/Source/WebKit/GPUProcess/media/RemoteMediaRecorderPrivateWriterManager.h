/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)

#include "MessageReceiver.h"
#include "RemoteMediaRecorderPrivateWriter.h"
#include "RemoteMediaRecorderPrivateWriterIdentifier.h"
#include "RemoteTrackInfo.h"
#include <WebCore/MediaSample.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
class Decoder;
class Encoder;

template<> struct ArgumentCoder<WebCore::MediaSamplesBlock::MediaSampleItem> {
    static void encode(Encoder&, const WebCore::MediaSamplesBlock::MediaSampleItem&);
    static std::optional<WebCore::MediaSamplesBlock::MediaSampleItem> decode(Decoder&);
};
}

namespace WebCore {
class SharedBuffer;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteMediaRecorderPrivateWriterProxy;
struct SharedPreferencesForWebProcess;

class RemoteMediaRecorderPrivateWriterManager : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaRecorderPrivateWriterManager);
    WTF_MAKE_NONCOPYABLE(RemoteMediaRecorderPrivateWriterManager);
public:
    RemoteMediaRecorderPrivateWriterManager(GPUConnectionToWebProcess&);
    ~RemoteMediaRecorderPrivateWriterManager();

    // IPC::MessageReceiver
    void ref() const final;
    void deref() const final;
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    bool allowsExitUnderMemoryPressure() { return m_remoteMediaRecorderPrivateWriters.isEmpty(); }

    // Messages.
    void create(RemoteMediaRecorderPrivateWriterIdentifier);
    void addMediaRecorderPrivateWriter(RemoteMediaRecorderPrivateWriterIdentifier);
    void addAudioTrack(RemoteMediaRecorderPrivateWriterIdentifier, RemoteAudioInfo, CompletionHandler<void(std::optional<uint8_t>)>&&);
    void addVideoTrack(RemoteMediaRecorderPrivateWriterIdentifier, RemoteVideoInfo, std::optional<CGAffineTransform>, CompletionHandler<void(std::optional<uint8_t>)>&&);
    void allTracksAdded(RemoteMediaRecorderPrivateWriterIdentifier, CompletionHandler<void(bool)>&&);
    using BlockPair = std::pair<WebCore::TrackInfo::TrackType, Vector<WebCore::MediaSamplesBlock::MediaSampleItem>>;
    void writeFrames(RemoteMediaRecorderPrivateWriterIdentifier, Vector<BlockPair>&&, const MediaTime&, CompletionHandler<void(Expected<Ref<WebCore::SharedBuffer>, WebCore::MediaRecorderPrivateWriter::Result>)>&&);
    void close(RemoteMediaRecorderPrivateWriterIdentifier, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>)>&&);

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    struct Writer {
        RefPtr<RemoteMediaRecorderPrivateWriterProxy> proxy;
        RefPtr<WebCore::AudioInfo> audioInfo;
        RefPtr<WebCore::VideoInfo> videoInfo;
    };

    HashMap<RemoteMediaRecorderPrivateWriterIdentifier, Writer> m_remoteMediaRecorderPrivateWriters;

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
