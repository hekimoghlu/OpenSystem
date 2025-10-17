/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

#include "HidConnection.h"
#include "VirtualAuthenticatorConfiguration.h"
#include <WebCore/FidoHidMessage.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
struct VirtualAuthenticatorConfiguration;
class VirtualAuthenticatorManager;

class VirtualHidConnection final : public HidConnection {
public:
    static Ref<VirtualHidConnection> create(const String& authenticatorId, const VirtualAuthenticatorConfiguration&, const WeakPtr<VirtualAuthenticatorManager>&);
    virtual ~VirtualHidConnection() = default;

private:
    explicit VirtualHidConnection(const String& authenticatorId, const VirtualAuthenticatorConfiguration&, const WeakPtr<VirtualAuthenticatorManager>&);

    void initialize() final;
    void terminate() final;
    DataSent sendSync(const Vector<uint8_t>& data) final;
    void send(Vector<uint8_t>&& data, DataSentCallback&&) final;
    void assembleRequest(Vector<uint8_t>&&);
    void parseRequest();

    void receiveHidMessage(fido::FidoHidMessage&&);
    void recieveResponseCode(fido::CtapDeviceResponseCode);

    WeakPtr<VirtualAuthenticatorManager> m_manager;
    VirtualAuthenticatorConfiguration m_configuration;
    std::optional<fido::FidoHidMessage> m_requestMessage;
    Vector<uint8_t> m_nonce;
    uint32_t m_currentChannel { fido::kHidBroadcastChannel };
    String m_authenticatorId;
};
} // namespace WebKit
#endif
