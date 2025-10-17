/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#if ENABLE(WEB_AUTHN)

#include "FidoService.h"
#include <wtf/RunLoop.h>

OBJC_CLASS NFReaderSession;

namespace WebKit {

class NfcConnection;

class NfcService : public FidoService {
public:
    static Ref<NfcService> create(AuthenticatorTransportServiceObserver&);
    ~NfcService();

    static bool isAvailable();

    // For NfcConnection.
    void didConnectTag();
    void didDetectMultipleTags() const;

protected:
    explicit NfcService(AuthenticatorTransportServiceObserver&);

#if HAVE(NEAR_FIELD)
    void setConnection(Ref<NfcConnection>&&); // For MockNfcConnection
#endif

private:
    void startDiscoveryInternal() final;
    void restartDiscoveryInternal() final;

    // Overrided by MockNfcService.
    virtual void platformStartDiscovery();

#if HAVE(NEAR_FIELD)
    // Only one reader session is allowed per time.
    // Keep the reader session alive here when it tries to connect to a tag.
    RefPtr<NfcConnection> m_connection;
#endif
    RunLoop::Timer m_restartTimer;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
