/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include "RemoteMediaEngineConfigurationFactoryProxy.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "SharedPreferencesForWebProcess.h"
#include <WebCore/MediaCapabilitiesDecodingInfo.h>
#include <WebCore/MediaCapabilitiesEncodingInfo.h>
#include <WebCore/MediaDecodingConfiguration.h>
#include <WebCore/MediaEncodingConfiguration.h>
#include <WebCore/MediaEngineConfigurationFactory.h>
#include <wtf/Algorithms.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaEngineConfigurationFactoryProxy);

RemoteMediaEngineConfigurationFactoryProxy::RemoteMediaEngineConfigurationFactoryProxy(GPUConnectionToWebProcess& connection)
    : m_connection(connection)
{
}

RemoteMediaEngineConfigurationFactoryProxy::~RemoteMediaEngineConfigurationFactoryProxy() = default;

void RemoteMediaEngineConfigurationFactoryProxy:: createDecodingConfiguration(WebCore::MediaDecodingConfiguration&& configuration, CompletionHandler<void(WebCore::MediaCapabilitiesDecodingInfo&&)>&& completion)
{
    MediaEngineConfigurationFactory::createDecodingConfiguration(WTFMove(configuration), [completion = WTFMove(completion)] (auto info) mutable {
        completion(WTFMove(info));
    });
}

void RemoteMediaEngineConfigurationFactoryProxy::createEncodingConfiguration(WebCore::MediaEncodingConfiguration&& configuration, CompletionHandler<void(WebCore::MediaCapabilitiesEncodingInfo&&)>&& completion)
{
    MediaEngineConfigurationFactory::createEncodingConfiguration(WTFMove(configuration), [completion = WTFMove(completion)] (auto info) mutable {
        completion(WTFMove(info));
    });
}

void RemoteMediaEngineConfigurationFactoryProxy::ref() const
{
    m_connection.get()->ref();
}

void RemoteMediaEngineConfigurationFactoryProxy::deref() const
{
    m_connection.get()->deref();
}

std::optional<SharedPreferencesForWebProcess> RemoteMediaEngineConfigurationFactoryProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr connection = m_connection.get())
        return connection->sharedPreferencesForWebProcess();
    return std::nullopt;
}

}

#endif
