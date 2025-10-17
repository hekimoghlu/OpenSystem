/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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

#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct MockWebAuthenticationConfiguration {
    enum class HidStage : bool {
        Info,
        Request
    };

    enum class HidSubStage : bool {
        Init,
        Msg
    };

    enum class HidError : uint8_t {
        Success,
        DataNotSent,
        EmptyReport,
        WrongChannelId,
        MaliciousPayload,
        UnsupportedOptions,
        WrongNonce
    };

    enum class NfcError : uint8_t {
        Success,
        NoTags,
        WrongTagType,
        NoConnections,
        MaliciousPayload
    };

    enum class UserVerification : uint8_t {
        No,
        Yes,
        Cancel,
        Presence
    };

    struct LocalConfiguration {
        UserVerification userVerification { UserVerification::No };
        bool acceptAttestation { false };
        String privateKeyBase64;
        String userCertificateBase64;
        String intermediateCACertificateBase64;
        String preferredCredentialIdBase64;
    };

    struct HidConfiguration {
        Vector<String> payloadBase64;
        HidStage stage { HidStage::Info };
        HidSubStage subStage { HidSubStage::Init };
        HidError error { HidError::Success };
        bool isU2f { false };
        bool keepAlive { false };
        bool fastDataArrival { false };
        bool continueAfterErrorData { false };
        bool canDowngrade { false };
        bool expectCancel { false };
        bool supportClientPin { false };
        bool supportInternalUV { false };
    };

    struct NfcConfiguration {
        NfcError error { NfcError::Success };
        Vector<String> payloadBase64;
        bool multipleTags { false };
        bool multiplePhysicalTags { false };
    };

    struct CcidConfiguration {
        Vector<String> payloadBase64;
    };

    bool silentFailure { false };
    std::optional<LocalConfiguration> local;
    std::optional<HidConfiguration> hid;
    std::optional<NfcConfiguration> nfc;
    std::optional<CcidConfiguration> ccid;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
