/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#include <WebCore/FidoHidMessage.h>
#include <WebCore/MockWebAuthenticationConfiguration.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

// The following basically simulates an external HID token that:
//    1. Supports only one protocol, either CTAP2 or U2F.
//    2. Doesn't support resident keys,
//    3. Doesn't support user verification.
// There are four stages for each CTAP request:
// FSM: Info::Init => Info::Msg => Request::Init => Request::Msg
// There are indefinite stages for each U2F request:
// FSM: Info::Init => Info::Msg => [Request::Init => Request::Msg]+
// According to different combinations of error and stages, error will manifest differently.
class MockHidConnection final : public HidConnection {
public:
    static Ref<MockHidConnection> create(IOHIDDeviceRef, const WebCore::MockWebAuthenticationConfiguration&);
    virtual ~MockHidConnection() = default;

private:
    MockHidConnection(IOHIDDeviceRef, const WebCore::MockWebAuthenticationConfiguration&);

    // HidConnection
    void initialize() final;
    void terminate() final;
    DataSent sendSync(const Vector<uint8_t>& data) final;
    void send(Vector<uint8_t>&& data, DataSentCallback&&) final;
    void registerDataReceivedCallbackInternal() final;

    void assembleRequest(Vector<uint8_t>&&);
    void parseRequest();
    void feedReports();
    bool stagesMatch() const;
    void shouldContinueFeedReports();
    void continueFeedReports();

    WebCore::MockWebAuthenticationConfiguration m_configuration;
    std::optional<fido::FidoHidMessage> m_requestMessage;
    WebCore::MockWebAuthenticationConfiguration::HidStage m_stage { WebCore::MockWebAuthenticationConfiguration::HidStage::Info };
    WebCore::MockWebAuthenticationConfiguration::HidSubStage m_subStage { WebCore::MockWebAuthenticationConfiguration::HidSubStage::Init };
    uint32_t m_currentChannel { fido::kHidBroadcastChannel };
    bool m_requireResidentKey { false };
    bool m_requireUserVerification  { false };
    Vector<uint8_t> m_nonce;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
