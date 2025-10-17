/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "NetworkProcessMain.h"

#include "AuxiliaryProcessMain.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include <WebCore/NetworkStorageSession.h>

#if USE(GCRYPT)
#include <pal/crypto/gcrypt/Initialization.h>
#endif

namespace WebKit {

class NetworkProcessMainSoup final: public AuxiliaryProcessMainBaseNoSingleton<NetworkProcess> {
public:
    bool platformInitialize() override
    {
#if USE(GCRYPT)
        PAL::GCrypt::initialize();
#endif
        return true;
    }

    void platformFinalize() override
    {
        // Needed to destroy the SoupSession and SoupCookieJar, e.g. to avoid
        // leaking SQLite temporary journaling files.
        Vector<PAL::SessionID> sessionIDs;
        process().forEachNetworkSession([&sessionIDs](auto& session) {
            sessionIDs.append(session.sessionID());
        });
        for (auto& sessionID : sessionIDs)
            process().destroySession(sessionID);
    }
};

int NetworkProcessMain(int argc, char** argv)
{
    return AuxiliaryProcessMain<NetworkProcessMainSoup>(argc, argv);
}

} // namespace WebKit
