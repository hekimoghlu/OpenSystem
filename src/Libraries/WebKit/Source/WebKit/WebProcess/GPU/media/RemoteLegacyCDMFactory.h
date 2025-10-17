/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "RemoteLegacyCDMIdentifier.h"
#include "RemoteLegacyCDMSessionIdentifier.h"
#include "WebProcessSupplement.h"
#include <WebCore/LegacyCDM.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class Settings;
}

namespace WebCore {
class CDMPrivateInterface;
class LegacyCDM;
}

namespace WebKit {

class GPUProcessConnection;
class RemoteLegacyCDM;
class RemoteLegacyCDMSession;
class WebProcess;


class RemoteLegacyCDMFactory final
    : public WebProcessSupplement
    , public CanMakeWeakPtr<RemoteLegacyCDMFactory> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLegacyCDMFactory);
public:
    explicit RemoteLegacyCDMFactory(WebProcess&);
    virtual ~RemoteLegacyCDMFactory();

    void registerFactory();

    static ASCIILiteral supplementName();

    GPUProcessConnection& gpuProcessConnection();

    void addSession(RemoteLegacyCDMSessionIdentifier, RemoteLegacyCDMSession&);
    void removeSession(RemoteLegacyCDMSessionIdentifier);

    RemoteLegacyCDM* findCDM(WebCore::CDMPrivateInterface*) const;

    void ref() const;
    void deref() const;

private:
    bool supportsKeySystem(const String&);
    bool supportsKeySystemAndMimeType(const String&, const String&);
    std::unique_ptr<WebCore::CDMPrivateInterface> createCDM(WebCore::LegacyCDM&);

    WeakRef<WebProcess> m_webProcess;
    HashMap<RemoteLegacyCDMSessionIdentifier, WeakPtr<RemoteLegacyCDMSession>> m_sessions;
    HashMap<RemoteLegacyCDMIdentifier, WeakPtr<RemoteLegacyCDM>> m_cdms;
    HashMap<String, bool> m_supportsKeySystemCache;
    HashMap<std::pair<String, String>, bool> m_supportsKeySystemAndMimeTypeCache;
};

}

#endif
