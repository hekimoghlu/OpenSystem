/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#include "RemoteLegacyCDMIdentifier.h"
#include "RemoteLegacyCDMSessionIdentifier.h"
#include "WebProcessSupplement.h"
#include <WebCore/MediaEngineConfigurationFactory.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class GPUProcessConnection;
class WebProcess;

class RemoteMediaEngineConfigurationFactory final
    : public WebProcessSupplement
    , public CanMakeWeakPtr<RemoteMediaEngineConfigurationFactory> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaEngineConfigurationFactory);
public:
    explicit RemoteMediaEngineConfigurationFactory(WebProcess&);
    virtual ~RemoteMediaEngineConfigurationFactory();

    // This is a supplement to WebProcess, which is a singleton.
    void ref() const { }
    void deref() const { }

    void registerFactory();

    static ASCIILiteral supplementName();

    GPUProcessConnection& gpuProcessConnection();

    void didReceiveSessionMessage(IPC::Connection&, IPC::Decoder&);

private:
    void createDecodingConfiguration(WebCore::MediaDecodingConfiguration&&, WebCore::MediaEngineConfigurationFactory::DecodingConfigurationCallback&&);
    void createEncodingConfiguration(WebCore::MediaEncodingConfiguration&&, WebCore::MediaEngineConfigurationFactory::EncodingConfigurationCallback&&);

    WeakRef<WebProcess> m_webProcess;
};

}

#endif
