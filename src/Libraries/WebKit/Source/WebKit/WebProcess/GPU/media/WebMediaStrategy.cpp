/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#include "WebMediaStrategy.h"

#include "GPUConnectionToWebProcess.h"
#include "GPUProcessConnection.h"
#include "RemoteAudioDestinationProxy.h"
#include "RemoteCDMFactory.h"
#include "WebProcess.h"
#include <WebCore/AudioDestination.h>
#include <WebCore/AudioIOCallback.h>
#include <WebCore/CDMFactory.h>
#include <WebCore/MediaPlayer.h>
#include <WebCore/NowPlayingManager.h>
#include <WebCore/SharedAudioDestination.h>

#if PLATFORM(COCOA)
#include "RemoteMediaRecorderPrivateWriter.h"
#include <WebCore/MediaSessionManagerCocoa.h>
#endif

#if ENABLE(MEDIA_SOURCE)
#include <WebCore/DeprecatedGlobalSettings.h>
#endif

namespace WebKit {

WebMediaStrategy::~WebMediaStrategy() = default;

#if ENABLE(WEB_AUDIO)
Ref<WebCore::AudioDestination> WebMediaStrategy::createAudioDestination(WebCore::AudioIOCallback& callback, const String& inputDeviceId,
    unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate)
{
    ASSERT(isMainRunLoop());
#if ENABLE(GPU_PROCESS)
    if (m_useGPUProcess)
        return WebCore::SharedAudioDestination::create(callback, numberOfOutputChannels, sampleRate, [inputDeviceId, numberOfInputChannels, numberOfOutputChannels, sampleRate] (WebCore::AudioIOCallback& callback) {
            return RemoteAudioDestinationProxy::create(callback, inputDeviceId, numberOfInputChannels, numberOfOutputChannels, sampleRate);
        });
#endif
    return WebCore::AudioDestination::create(callback, inputDeviceId, numberOfInputChannels, numberOfOutputChannels, sampleRate);
}
#endif

std::unique_ptr<WebCore::NowPlayingManager> WebMediaStrategy::createNowPlayingManager() const
{
    ASSERT(isMainRunLoop());
#if ENABLE(GPU_PROCESS)
    if (m_useGPUProcess) {
        class NowPlayingInfoForGPUManager : public WebCore::NowPlayingManager {
            void clearNowPlayingInfoPrivate() final
            {
                if (RefPtr connection = WebProcess::singleton().existingGPUProcessConnection())
                    connection->connection().send(Messages::GPUConnectionToWebProcess::ClearNowPlayingInfo { }, 0);
            }

            void setNowPlayingInfoPrivate(const WebCore::NowPlayingInfo& nowPlayingInfo, bool) final
            {
                Ref connection = WebProcess::singleton().ensureGPUProcessConnection().connection();
                connection->send(Messages::GPUConnectionToWebProcess::SetNowPlayingInfo { nowPlayingInfo }, 0);
            }
        };
        return makeUnique<NowPlayingInfoForGPUManager>();
    }
#endif
    return WebCore::MediaStrategy::createNowPlayingManager();
}

bool WebMediaStrategy::hasThreadSafeMediaSourceSupport() const
{
#if ENABLE(GPU_PROCESS)
    return m_useGPUProcess;
#else
    return false;
#endif
}

#if ENABLE(MEDIA_SOURCE)
void WebMediaStrategy::enableMockMediaSource()
{
    ASSERT(isMainRunLoop());
#if USE(AVFOUNDATION)
    WebCore::DeprecatedGlobalSettings::setAVFoundationEnabled(false);
#endif
#if USE(GSTREAMER)
    WebCore::DeprecatedGlobalSettings::setGStreamerEnabled(false);
#endif
    m_mockMediaSourceEnabled = true;
#if ENABLE(GPU_PROCESS)
    if (m_useGPUProcess) {
        Ref connection = WebProcess::singleton().ensureGPUProcessConnection().connection();
        connection->send(Messages::GPUConnectionToWebProcess::EnableMockMediaSource { }, 0);
        return;
    }
#endif
    WebCore::MediaStrategy::addMockMediaSourceEngine();
}
#endif

#if PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)
std::unique_ptr<MediaRecorderPrivateWriter> WebMediaStrategy::createMediaRecorderPrivateWriter(MediaRecorderContainerType type, WebCore::MediaRecorderPrivateWriterListener& listener) const
{
    ASSERT(isMainRunLoop());
#if ENABLE(GPU_PROCESS)
    if (type != MediaRecorderContainerType::Mp4)
        return nullptr;
    if (m_useGPUProcess)
        return RemoteMediaRecorderPrivateWriter::create(WebProcess::singleton().ensureGPUProcessConnection(), listener);
#else
    UNUSED_PARAM(type);
    UNUSED_PARAM(listener);
#endif
    return nullptr;
}
#endif
} // namespace WebKit
