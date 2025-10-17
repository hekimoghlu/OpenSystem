/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#include "RemoteAudioSourceProvider.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)

#include "GPUProcessConnection.h"
#include "Logging.h"
#include "RemoteAudioSourceProviderManager.h"
#include "RemoteMediaPlayerProxyMessages.h"

namespace WebCore {
#if !RELEASE_LOG_DISABLED
extern WTFLogChannel LogMedia;
#endif
}

namespace WebKit {
using namespace WebCore;

Ref<RemoteAudioSourceProvider> RemoteAudioSourceProvider::create(WebCore::MediaPlayerIdentifier identifier, WTF::LoggerHelper& helper)
{
    auto provider = adoptRef(*new RemoteAudioSourceProvider(identifier, helper));
    provider->m_gpuProcessConnection.get()->audioSourceProviderManager().addProvider(provider.copyRef());
    return provider;
}

RemoteAudioSourceProvider::RemoteAudioSourceProvider(MediaPlayerIdentifier identifier, WTF::LoggerHelper& helper)
    : m_identifier(identifier)
    , m_gpuProcessConnection(WebProcess::singleton().ensureGPUProcessConnection())
#if !RELEASE_LOG_DISABLED
    , m_logger(helper.logger())
    , m_logIdentifier(helper.logIdentifier())
#endif
{
    ASSERT(isMainRunLoop());
    UNUSED_PARAM(helper);

#if ENABLE(WEB_AUDIO)
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    gpuProcessConnection->connection().send(Messages::RemoteMediaPlayerProxy::CreateAudioSourceProvider { }, identifier);
#endif
}

RemoteAudioSourceProvider::~RemoteAudioSourceProvider()
{
}

void RemoteAudioSourceProvider::close()
{
    ASSERT(isMainRunLoop());
    if (auto gpuProcessConnection = m_gpuProcessConnection.get())
        gpuProcessConnection->audioSourceProviderManager().removeProvider(m_identifier);
}

void RemoteAudioSourceProvider::hasNewClient(AudioSourceProviderClient* client)
{
    if (auto gpuProcessConnection = m_gpuProcessConnection.get())
        gpuProcessConnection->connection().send(Messages::RemoteMediaPlayerProxy::SetShouldEnableAudioSourceProvider { !!client }, m_identifier);
}

void RemoteAudioSourceProvider::audioSamplesAvailable(const PlatformAudioData& data, const AudioStreamDescription& description, size_t size)
{
    receivedNewAudioSamples(data, description, size);
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& RemoteAudioSourceProvider::logChannel() const
{
    return JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, Media);
}
#endif

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)
