/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "GPUProcessConnection.h"
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/WebAudioSourceProviderCocoa.h>
#include <wtf/LoggerHelper.h>

namespace WebKit {

class RemoteAudioSourceProvider final
    : public WebCore::WebAudioSourceProviderCocoa
#if !RELEASE_LOG_DISABLED
    , protected WTF::LoggerHelper
#endif
{
public:
    static Ref<RemoteAudioSourceProvider> create(WebCore::MediaPlayerIdentifier, WTF::LoggerHelper&);
    ~RemoteAudioSourceProvider();

    void audioSamplesAvailable(const WebCore::PlatformAudioData&, const WebCore::AudioStreamDescription&, size_t);
    void close();

    WebCore::MediaPlayerIdentifier identifier() const { return m_identifier; }

private:
    RemoteAudioSourceProvider(WebCore::MediaPlayerIdentifier, WTF::LoggerHelper&);

    // WebCore::WebAudioSourceProviderCocoa
    void hasNewClient(WebCore::AudioSourceProviderClient*) final;

#if !RELEASE_LOG_DISABLED
    WTF::LoggerHelper& loggerHelper() final { return *this; }

    // WTF::LoggerHelper
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "RemoteAudioSourceProvider"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    WebCore::MediaPlayerIdentifier m_identifier;
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO) && PLATFORM(COCOA)
