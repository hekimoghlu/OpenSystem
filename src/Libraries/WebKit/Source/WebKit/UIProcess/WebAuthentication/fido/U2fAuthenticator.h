/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include "FidoAuthenticator.h"
#include <wtf/RunLoop.h>

namespace apdu {
class ApduResponse;
}

namespace WebKit {

class CtapDriver;

class U2fAuthenticator final : public FidoAuthenticator {
public:
    static Ref<U2fAuthenticator> create(Ref<CtapDriver>&& driver)
    {
        return adoptRef(*new U2fAuthenticator(WTFMove(driver)));
    }

private:
    explicit U2fAuthenticator(Ref<CtapDriver>&&);

    void makeCredential() final;
    void checkExcludeList(size_t index);
    void issueRegisterCommand();
    void getAssertion() final;
    void issueSignCommand(size_t index);

    enum class CommandType : uint8_t {
        RegisterCommand,
        CheckOnlyCommand,
        BogusCommandExcludeCredentialsMatch,
        BogusCommandNoCredentials,
        SignCommand
    };
    void issueNewCommand(Vector<uint8_t>&& command, CommandType);
    void retryLastCommand() { issueCommand(m_lastCommand, m_lastCommandType); }
    void issueCommand(const Vector<uint8_t>& command, CommandType);
    void responseReceived(Vector<uint8_t>&& response, CommandType);
    void continueRegisterCommandAfterResponseReceived(apdu::ApduResponse&&);
    void continueCheckOnlyCommandAfterResponseReceived(apdu::ApduResponse&&);
    void continueBogusCommandExcludeCredentialsMatchAfterResponseReceived(apdu::ApduResponse&&);
    void continueBogusCommandNoCredentialsAfterResponseReceived(apdu::ApduResponse&&);
    void continueSignCommandAfterResponseReceived(apdu::ApduResponse&&);

    RunLoop::Timer m_retryTimer;
    Vector<uint8_t> m_lastCommand;
    CommandType m_lastCommandType;
    size_t m_nextListIndex { 0 };
    bool m_isAppId { false };
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
