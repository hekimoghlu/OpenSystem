/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

#include "PrivateClickMeasurementClient.h"
#include <pal/SessionID.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class NetworkSession;
class NetworkProcess;

namespace PCM {

class ClientImpl : public Client {
    WTF_MAKE_TZONE_ALLOCATED(ClientImpl);
public:
    ClientImpl(NetworkSession&, NetworkProcess&);

private:
    void broadcastConsoleMessage(JSC::MessageLevel, const String&) final;
    bool featureEnabled() const final;
    bool debugModeEnabled() const final;
    bool usesEphemeralDataStore() const final;
    bool runningInDaemon() const final { return false; }

    WeakPtr<NetworkSession> m_networkSession;
    Ref<NetworkProcess> m_networkProcess;
};

} // namespace PCM

} // namespace WebKit
