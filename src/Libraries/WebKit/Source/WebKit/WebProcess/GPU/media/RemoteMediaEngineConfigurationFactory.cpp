/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#include "RemoteMediaEngineConfigurationFactory.h"

#if ENABLE(GPU_PROCESS)

#include "GPUProcessConnection.h"
#include "RemoteMediaEngineConfigurationFactoryProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/MediaCapabilitiesDecodingInfo.h>
#include <WebCore/MediaCapabilitiesEncodingInfo.h>
#include <WebCore/MediaDecodingConfiguration.h>
#include <WebCore/MediaEncodingConfiguration.h>
#include <WebCore/MediaEngineConfigurationFactory.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaEngineConfigurationFactory);

RemoteMediaEngineConfigurationFactory::RemoteMediaEngineConfigurationFactory(WebProcess& webProcess)
    : m_webProcess(webProcess)
{
}

RemoteMediaEngineConfigurationFactory::~RemoteMediaEngineConfigurationFactory() = default;

void RemoteMediaEngineConfigurationFactory::registerFactory()
{
    MediaEngineConfigurationFactory::clearFactories();

    auto createDecodingConfiguration = [weakThis = WeakPtr { *this }] (MediaDecodingConfiguration&& configuration, MediaEngineConfigurationFactory::DecodingConfigurationCallback&& callback) {
        if (!weakThis) {
            callback({{ }, WTFMove(configuration)});
            return;
        }

        weakThis->createDecodingConfiguration(WTFMove(configuration), WTFMove(callback));
    };

#if PLATFORM(COCOA)
    MediaEngineConfigurationFactory::CreateEncodingConfiguration createEncodingConfiguration = nullptr;
#else
    auto createEncodingConfiguration = [weakThis = WeakPtr { *this }] (MediaEncodingConfiguration&& configuration, MediaEngineConfigurationFactory::EncodingConfigurationCallback&& callback) {
        if (!weakThis) {
            callback({{ }, WTFMove(configuration)});
            return;
        }

        weakThis->createEncodingConfiguration(WTFMove(configuration), WTFMove(callback));
    };
#endif

    MediaEngineConfigurationFactory::installFactory({ WTFMove(createDecodingConfiguration), WTFMove(createEncodingConfiguration) });
}

ASCIILiteral RemoteMediaEngineConfigurationFactory::supplementName()
{
    return "RemoteMediaEngineConfigurationFactory"_s;
}

GPUProcessConnection& RemoteMediaEngineConfigurationFactory::gpuProcessConnection()
{
    return WebProcess::singleton().ensureGPUProcessConnection();
}

void RemoteMediaEngineConfigurationFactory::createDecodingConfiguration(MediaDecodingConfiguration&& configuration, MediaEngineConfigurationFactory::DecodingConfigurationCallback&& callback)
{
    if (!m_webProcess->mediaPlaybackEnabled())
        return callback({ });

    gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteMediaEngineConfigurationFactoryProxy::CreateDecodingConfiguration(WTFMove(configuration)), [callback = WTFMove(callback)] (MediaCapabilitiesDecodingInfo&& info) mutable {
        callback(WTFMove(info));
    });
}

void RemoteMediaEngineConfigurationFactory::createEncodingConfiguration(MediaEncodingConfiguration&& configuration, MediaEngineConfigurationFactory::EncodingConfigurationCallback&& callback)
{
    if (!m_webProcess->mediaPlaybackEnabled())
        return callback({ });

    gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteMediaEngineConfigurationFactoryProxy::CreateEncodingConfiguration(WTFMove(configuration)), [callback = WTFMove(callback)] (MediaCapabilitiesEncodingInfo&& info) mutable {
        callback(WTFMove(info));
    });
}

}

#endif // ENABLE(GPU_PROCESS)
