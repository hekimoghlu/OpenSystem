/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "RemoteCDMIdentifier.h"
#include "RemoteCDMInstanceIdentifier.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include "WebProcessSupplement.h"
#include <WebCore/CDMFactory.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class Settings;
}

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class GPUProcessConnection;
class RemoteCDM;
class RemoteCDMInstanceSession;
class WebProcess;

class RemoteCDMFactory final
    : public WebCore::CDMFactory
    , public WebProcessSupplement
    , public CanMakeWeakPtr<RemoteCDMFactory> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCDMFactory);
public:
    explicit RemoteCDMFactory(WebProcess&);
    virtual ~RemoteCDMFactory();

    void ref() const;
    void deref() const;

    static ASCIILiteral supplementName();

    GPUProcessConnection& gpuProcessConnection();

    void registerFactory(Vector<WebCore::CDMFactory*>&);

    void didReceiveSessionMessage(IPC::Connection&, IPC::Decoder&);

    void addSession(RemoteCDMInstanceSession&);
    void removeSession(RemoteCDMInstanceSessionIdentifier);

    void removeInstance(RemoteCDMInstanceIdentifier);

private:
    std::unique_ptr<WebCore::CDMPrivate> createCDM(const String&, const WebCore::CDMPrivateClient&) final;
    bool supportsKeySystem(const String&) final;

    WeakRef<WebProcess> m_webProcess;
    HashMap<RemoteCDMInstanceSessionIdentifier, WeakPtr<RemoteCDMInstanceSession>> m_sessions;
    HashMap<RemoteCDMIdentifier, std::unique_ptr<RemoteCDM>> m_cdms;
};

}

#endif
