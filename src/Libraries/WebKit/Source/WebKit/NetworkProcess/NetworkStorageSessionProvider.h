/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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

#include "NetworkProcess.h"
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/StorageSessionProvider.h>
#include <pal/SessionID.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class NetworkStorageSessionProvider final : public WebCore::StorageSessionProvider {
public:
    static Ref<NetworkStorageSessionProvider> create(NetworkProcess& networkProcess, PAL::SessionID sessionID) { return adoptRef(*new NetworkStorageSessionProvider(networkProcess, sessionID)); }
    
private:
    NetworkStorageSessionProvider(NetworkProcess& networkProcess, PAL::SessionID sessionID)
        : m_networkProcess(networkProcess)
        , m_sessionID(sessionID) { }

    WebCore::NetworkStorageSession* storageSession() const final
    {
        if (m_networkProcess)
            return m_networkProcess->storageSession(m_sessionID);
        return nullptr;
    }

    WeakPtr<NetworkProcess> m_networkProcess;
    PAL::SessionID m_sessionID;
};

}
