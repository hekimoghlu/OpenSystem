/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#include "CtapNfcDriver.h"

#if ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)

#include "Logging.h"
#include <WebCore/ApduCommand.h>
#include <WebCore/ApduResponse.h>
#include <wtf/Assertions.h>
#include <wtf/RunLoop.h>

namespace WebKit {
using namespace apdu;
using namespace fido;

Ref<CtapNfcDriver> CtapNfcDriver::create(Ref<NfcConnection>&& connection)
{
    return adoptRef(*new CtapNfcDriver(WTFMove(connection)));
}

CtapNfcDriver::CtapNfcDriver(Ref<NfcConnection>&& connection)
    : CtapDriver(WebCore::AuthenticatorTransport::Nfc)
    , m_connection(WTFMove(connection))
{
}

// FIXME(200934): Support NFCCTAP_GETRESPONSE
void CtapNfcDriver::transact(Vector<uint8_t>&& data, ResponseCallback&& callback)
{
    // For CTAP2, commands follow:
    // https://fidoalliance.org/specs/fido-v2.0-ps-20190130/fido-client-to-authenticator-protocol-v2.0-ps-20190130.html#nfc-command-framing
    if (isCtap2Protocol()) {
        if (!isValidSize(data.size()))
            RELEASE_LOG(WebAuthn, "CtapNfcDriver::transact Sending data larger than maxSize. msgSize=%ld", data.size());
        ApduCommand command;
        command.setCla(kCtapNfcApduCla);
        command.setIns(kCtapNfcApduIns);
        command.setData(WTFMove(data));
        command.setResponseLength(ApduCommand::kApduMaxResponseLength);

        auto apduResponse = ApduResponse::createFromMessage(m_connection->transact(command.getEncodedCommand()));
        if (!apduResponse) {
            respondAsync(WTFMove(callback), { });
            return;
        }
        if (apduResponse->status() == ApduResponse::Status::SW_INS_NOT_SUPPORTED) {
            // Return kCtap1ErrInvalidCommand instead of an empty response to signal FidoService to create a U2F authenticator
            // for the getInfo stage.
            respondAsync(WTFMove(callback), { static_cast<uint8_t>(CtapDeviceResponseCode::kCtap1ErrInvalidCommand) });
            return;
        }
        if (apduResponse->status() != ApduResponse::Status::SW_NO_ERROR) {
            respondAsync(WTFMove(callback), { });
            return;
        }

        respondAsync(WTFMove(callback), WTFMove(apduResponse->data()));
        return;
    }


    // For U2F, U2fAuthenticator would handle the APDU encoding.
    // https://fidoalliance.org/specs/fido-u2f-v1.2-ps-20170411/fido-u2f-nfc-protocol-v1.2-ps-20170411.html#framing
    callback(m_connection->transact(WTFMove(data)));
}

// Return the response async to match the HID behaviour, such that nfc could fit into the current infra.
void CtapNfcDriver::respondAsync(ResponseCallback&& callback, Vector<uint8_t>&& response) const
{
    RunLoop::main().dispatch([callback = WTFMove(callback), response = WTFMove(response)] () mutable {
        callback(WTFMove(response));
    });
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)
