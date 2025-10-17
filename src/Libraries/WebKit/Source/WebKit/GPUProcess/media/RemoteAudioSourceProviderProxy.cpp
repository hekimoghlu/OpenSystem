/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "RemoteAudioSourceProviderProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)

#include "RemoteAudioSourceProviderManagerMessages.h"
#include <WebCore/AudioSourceProviderAVFObjC.h>

namespace WebKit {

Ref<RemoteAudioSourceProviderProxy> RemoteAudioSourceProviderProxy::create(WebCore::MediaPlayerIdentifier identifier, Ref<IPC::Connection>&& connection, WebCore::AudioSourceProviderAVFObjC& localProvider)
{
    auto remoteProvider = adoptRef(*new RemoteAudioSourceProviderProxy(identifier, WTFMove(connection)));

    localProvider.setConfigureAudioStorageCallback([remoteProvider](auto&&... args) {
        return remoteProvider->configureAudioStorage(args...);
    });
    localProvider.setAudioCallback([remoteProvider](auto startFrame, auto numberOfFrames) {
        remoteProvider->newAudioSamples(startFrame, numberOfFrames);
    });

    return remoteProvider;
}

RemoteAudioSourceProviderProxy::RemoteAudioSourceProviderProxy(WebCore::MediaPlayerIdentifier identifier, Ref<IPC::Connection>&& connection)
    : m_identifier(identifier)
    , m_connection(WTFMove(connection))
{
}

RemoteAudioSourceProviderProxy::~RemoteAudioSourceProviderProxy() = default;

std::unique_ptr<WebCore::CARingBuffer> RemoteAudioSourceProviderProxy::configureAudioStorage(const WebCore::CAAudioStreamDescription& format, size_t frameCount)
{
    auto result = ProducerSharedCARingBuffer::allocate(format, frameCount);
    RELEASE_ASSERT(result); // FIXME(https://bugs.webkit.org/show_bug.cgi?id=262690): Handle allocation failure.
    auto [ringBuffer, handle] = WTFMove(*result);
    protectedConnection()->send(Messages::RemoteAudioSourceProviderManager::AudioStorageChanged { m_identifier, WTFMove(handle), format }, 0);
    // Use a redundant variable to avoid move in return position and to obtain copy elision. Clang or libc++ does not allow returning covariant of Ts from std::unique_ptr<T>s in this position.
    std::unique_ptr<WebCore::CARingBuffer> caRingBuffer = WTFMove(ringBuffer);  // NOLINT: see above.
    return caRingBuffer;
}

void RemoteAudioSourceProviderProxy::newAudioSamples(uint64_t startFrame, uint64_t numberOfFrames)
{
    protectedConnection()->send(Messages::RemoteAudioSourceProviderManager::AudioSamplesAvailable { m_identifier, startFrame, numberOfFrames }, 0);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)
