/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#if ENABLE(GPU_PROCESS)

#include "Connection.h"
#include "MessageReceiver.h"
#include <WebCore/MediaEngineConfigurationFactory.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class GPUConnectionToWebProcess;
struct SharedPreferencesForWebProcess;

class RemoteMediaEngineConfigurationFactoryProxy final : private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaEngineConfigurationFactoryProxy);
public:
    explicit RemoteMediaEngineConfigurationFactoryProxy(GPUConnectionToWebProcess&);
    virtual ~RemoteMediaEngineConfigurationFactoryProxy();

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }

    void ref() const final;
    void deref() const final;

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;
private:
    friend class GPUProcessConnection;
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void createDecodingConfiguration(WebCore::MediaDecodingConfiguration&&, CompletionHandler<void(WebCore::MediaCapabilitiesDecodingInfo&&)>&&);
    void createEncodingConfiguration(WebCore::MediaEncodingConfiguration&&, CompletionHandler<void(WebCore::MediaCapabilitiesEncodingInfo&&)>&&);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connection; // Cannot be null.
};

}

#endif
