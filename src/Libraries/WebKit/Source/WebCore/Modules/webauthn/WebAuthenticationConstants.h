/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

namespace COSE {

// See RFC 8152 - CBOR Object Signing and Encryption <https://tools.ietf.org/html/rfc8152>
// Labels
const int64_t alg = 3;
const int64_t crv = -1;
const int64_t kty = 1;
const int64_t x = -2;
const int64_t y = -3;

// Values
const int64_t EC2 = 2;
const int64_t ES256 = -7;
const int64_t RS256 = -257;
const int64_t ECDH256 = -25;
const int64_t P_256 = 1;

} // namespace COSE

namespace WebCore {

// Length of the SHA-256 hash of the RP ID asssociated with the credential:
// https://www.w3.org/TR/webauthn/#sec-authenticator-data
const size_t rpIdHashLength = 32;

// Length of the flags:
// https://www.w3.org/TR/webauthn/#sec-authenticator-data
const size_t flagsLength = 1;

// Length of the signature counter, 32-bit unsigned big-endian integer:
// https://www.w3.org/TR/webauthn/#sec-authenticator-data
const size_t signCounterLength = 4;

// Length of the AAGUID of the authenticator:
// https://www.w3.org/TR/webauthn/#sec-attested-credential-data
const size_t aaguidLength = 16;

// Length of the byte length L of Credential ID, 16-bit unsigned big-endian
// integer: https://www.w3.org/TR/webauthn/#sec-attested-credential-data
const size_t credentialIdLengthLength = 2;

// Per Section 2.3.5 of http://www.secg.org/sec1-v2.pdf
const size_t ES256FieldElementLength = 32;

// https://www.w3.org/TR/webauthn/#none-attestation
const char noneAttestationValue[] = "none";

// https://www.w3.org/TR/webauthn-1/#dom-collectedclientdata-type
enum class ClientDataType : bool {
    Create,
    Get
};

enum class ShouldZeroAAGUID : bool {
    No,
    Yes
};

#if defined(__OBJC__)
NSString * const LocalAuthenticatorAccessGroup = @"com.apple.webkit.webauthn";
#endif

// Credential serialization
constexpr const char privateKeyKey[] = "priv";
constexpr const char keyTypeKey[] = "key_type";
constexpr const char keySizeKey[] = "key_size";
constexpr const char relyingPartyKey[] = "rp";
constexpr const char applicationTagKey[] = "tag";

constexpr auto authenticatorTransportUsb = "usb"_s;
constexpr auto authenticatorTransportNfc = "nfc"_s;
constexpr auto authenticatorTransportBle = "ble"_s;
constexpr auto authenticatorTransportInternal = "internal"_s;
constexpr auto authenticatorTransportCable = "cable"_s;
constexpr auto authenticatorTransportSmartCard = "smart-card"_s;
constexpr auto authenticatorTransportHybrid = "hybrid"_s;


} // namespace WebCore

namespace WebAuthn {

enum class Scope {
    CrossOrigin,
    SameOrigin,
    SameSite
};

// https://www.w3.org/TR/webauthn-2/#authenticator-data
constexpr uint8_t userPresenceFlag = 0b00000001;
constexpr uint8_t userVerifiedFlag = 0b00000100;
constexpr uint8_t attestedCredentialDataIncludedFlag = 0b01000000;
// https://github.com/w3c/webauthn/pull/1695
constexpr uint8_t backupEligibilityFlag = 0b00001000;
constexpr uint8_t backupStateFlag = 0b00010000;

} // namespace WebAuthn
