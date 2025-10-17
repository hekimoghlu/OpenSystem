/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)

#include "Connection.h"
#include "SharedCARingBuffer.h"
#include <WebCore/AudioSourceProviderClient.h>
#include <WebCore/CAAudioStreamDescription.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class AudioSourceProviderAVFObjC;
class CARingBuffer;
}

namespace WebKit {

class RemoteAudioSourceProviderProxy : public ThreadSafeRefCounted<RemoteAudioSourceProviderProxy>
    , public WebCore::AudioSourceProviderClient {
public:
    static Ref<RemoteAudioSourceProviderProxy> create(WebCore::MediaPlayerIdentifier, Ref<IPC::Connection>&&, WebCore::AudioSourceProviderAVFObjC&);
    ~RemoteAudioSourceProviderProxy();

    void newAudioSamples(uint64_t startFrame, uint64_t endFrame);

private:
    RemoteAudioSourceProviderProxy(WebCore::MediaPlayerIdentifier, Ref<IPC::Connection>&&);
    std::unique_ptr<WebCore::CARingBuffer> configureAudioStorage(const WebCore::CAAudioStreamDescription&, size_t frameCount);

    // AudioSourceProviderClient
    void setFormat(size_t numberOfChannels, float sampleRate) final { }

    Ref<IPC::Connection> protectedConnection() const { return m_connection; }

    WebCore::MediaPlayerIdentifier m_identifier;
    Ref<IPC::Connection> m_connection;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)
