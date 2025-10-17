/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include "PushMessageCrypto.h"

#include "PushCrypto.h"
#include <array>
#include <wtf/ByteOrder.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore::PushCrypto {

// Arbitrary limit that's larger than the largest payload APNS should ever give us.
static constexpr size_t maxPushPayloadLength = 65535;

// From RFC8291.
static constexpr size_t saltLength = 16;
static constexpr size_t sharedAuthSecretLength = 16;

ClientKeys ClientKeys::generate()
{
    std::array<uint8_t, sharedAuthSecretLength> sharedAuthSecret;
    cryptographicallyRandomValues(sharedAuthSecret);

    return ClientKeys {
        P256DHKeyPair::generate(),
        Vector<uint8_t> { sharedAuthSecret }
    };
}

static bool areClientKeyLengthsValid(const ClientKeys& clientKeys)
{
    return clientKeys.clientP256DHKeyPair.publicKey.size() == p256dhPublicKeyLength && clientKeys.clientP256DHKeyPair.privateKey.size() == p256dhPrivateKeyLength && clientKeys.sharedAuthSecret.size() == sharedAuthSecretLength;
}

static size_t computeAES128GCMPaddingLength(std::span<const uint8_t> data)
{
    /*
     * Compute padding length as defined in RFC8188 Section 2:
     *
     *   +-----------+-----+
     *   |   data    | pad |
     *   +-----------+-----+
     *
     * pad must be of non-zero length and is a delimiter octet (0x02) followed by any number of 0x00 octets.
     */
    if (data.empty())
        return SIZE_MAX;

    size_t current = data.size() - 1;
    while (current > 0 && (data[current] == 0x00))
        --current;
    if (data[current] != 0x02)
        return SIZE_MAX;

    return data.size() - current;
}

std::optional<Vector<uint8_t>> decryptAES128GCMPayload(const ClientKeys& clientKeys, std::span<const uint8_t> payload)
{
    if (!areClientKeyLengthsValid(clientKeys))
        return std::nullopt;

    // Extract encryption parameters from header as described in RFC8188.
    struct PayloadHeader {
        std::array<uint8_t, saltLength> salt;
        std::array<uint8_t, 4> ignored;
        uint8_t keyLength;
        std::array<uint8_t, p256dhPublicKeyLength> serverPublicKey;
    };
    static_assert(sizeof(PayloadHeader) == 86);
    static constexpr size_t minPushPayloadLength = sizeof(PayloadHeader) + 1 /* minPaddingLength */ + aes128GCMTagLength;

    if (payload.size() < minPushPayloadLength || payload.size() > maxPushPayloadLength)
        return std::nullopt;

    PayloadHeader header;
    memcpySpan(asMutableByteSpan(header), payload.first(sizeof(header)));

    if (header.keyLength != p256dhPublicKeyLength)
        return std::nullopt;

    /*
     * The rest of the comments are snippets from RFC8291 3.4.
     *
     * -- For a user agent:
     * ecdh_secret = ECDH(ua_private, as_public)
     */
    auto ecdhSecretResult = computeP256DHSharedSecret(header.serverPublicKey, clientKeys.clientP256DHKeyPair);
    if (!ecdhSecretResult)
        return std::nullopt;

    /*
     * # HKDF-Extract(salt=auth_secret, IKM=ecdh_secret)
     * PRK_key = HMAC-SHA-256(auth_secret, ecdh_secret)
     */
    auto prkKey = hmacSHA256(clientKeys.sharedAuthSecret, *ecdhSecretResult);

    /*
     * # HKDF-Expand(PRK_key, key_info, L_key=32)
     * key_info = "WebPush: info" || 0x00 || ua_public || as_public
     * IKM = HMAC-SHA-256(PRK_key, key_info || 0x01)
     */
    struct KeyInfo {
        char label[14] = { "WebPush: info" };
        std::array<uint8_t, p256dhPublicKeyLength> clientKey;
        std::array<uint8_t, p256dhPublicKeyLength> serverKey;
        uint8_t end = 0x01;
    };
    static_assert(sizeof(KeyInfo) == 145);

    KeyInfo keyInfo;
    memcpySpan(std::span { keyInfo.clientKey }, clientKeys.clientP256DHKeyPair.publicKey.span().first(p256dhPublicKeyLength));
    memcpySpan(std::span { keyInfo.serverKey }, std::span { header.serverPublicKey });

    auto ikm = hmacSHA256(prkKey, asByteSpan(keyInfo));

    /*
     * # HKDF-Extract(salt, IKM)
     * PRK = HMAC-SHA-256(salt, IKM)
     */
    auto prk = hmacSHA256(header.salt, ikm);

    /*
     * # HKDF-Expand(PRK, cek_info, L_cek=16)
     * cek_info = "Content-Encoding: aes128gcm" || 0x00
     * CEK = HMAC-SHA-256(PRK, cek_info || 0x01)[0..15]
     */
    static const auto cekInfo = "Content-Encoding: aes128gcm\x00\x01"_span8;
    auto cek = hmacSHA256(prk, cekInfo);
    cek.shrink(16);

    /*
     * # HKDF-Expand(PRK, nonce_info, L_nonce=12)
     * nonce_info = "Content-Encoding: nonce" || 0x00
     * NONCE = HMAC-SHA-256(PRK, nonce_info || 0x01)[0..11]
     */
    static const auto nonceInfo = "Content-Encoding: nonce\x00\x01"_span8;
    auto nonce = hmacSHA256(prk, nonceInfo);
    nonce.shrink(12);

    // Finally, decrypt with AES128GCM and return the unpadded plaintext.
    auto cipherText = payload.subspan(sizeof(header));
    auto plainTextResult = decryptAES128GCM(cek, nonce, cipherText);
    if (!plainTextResult)
        return std::nullopt;

    auto plainText = WTFMove(plainTextResult.value());
    size_t paddingLength = computeAES128GCMPaddingLength(plainText.span());
    if (paddingLength == SIZE_MAX)
        return std::nullopt;

    plainText.shrink(plainText.size() - paddingLength);
    return plainText;
}

static size_t computeAESGCMPaddingLength(std::span<const uint8_t> span)
{
    /*
     * Compute padding length as defined in draft-ietf-httpbis-encryption-encoding-03:
     *
     *   +-----+-----------+
     *   | pad |   data    |
     *   +-----+-----------+
     *
     * Padding consists of a two octet unsigned integer in network byte order, followed by that
     * number of 0x00 octets. The minimum padding size is 2 bytes.
     */
    if (span.size() < 2)
        return SIZE_MAX;

    uint16_t paddingLength;
    memcpySpan(asMutableByteSpan(paddingLength), span.first(2));
    paddingLength = ntohs(paddingLength);

    size_t current = 2;
    uint16_t paddingLeft = paddingLength;
    while (current < span.size() && span[current] == 0x0 && paddingLeft) {
        ++current;
        --paddingLeft;
    }

    if (paddingLeft)
        return SIZE_MAX;

    return current;
}

std::optional<Vector<uint8_t>> decryptAESGCMPayload(const ClientKeys& clientKeys, std::span<const uint8_t> serverP256DHPublicKey, std::span<const uint8_t> salt, std::span<const uint8_t> payload)
{
    if (!areClientKeyLengthsValid(clientKeys) || serverP256DHPublicKey.size() != p256dhPublicKeyLength || salt.size() != saltLength)
        return std::nullopt;

    // Padding must be at least the size of the two octet unsigned integer used in the padding scheme plus the size of the AES128GCM tag.
    if (payload.size() < 2 + aes128GCMTagLength || payload.size() > maxPushPayloadLength)
        return std::nullopt;

    /*
     * These comments are snippets from draft-ietf-webpush-encryption-04.
     *
     * -- For a User Agent:
     * ecdh_secret = ECDH(ua_private, as_public)
     */
    auto ecdhSecretResult = computeP256DHSharedSecret(serverP256DHPublicKey, clientKeys.clientP256DHKeyPair);
    if (!ecdhSecretResult)
        return std::nullopt;

    /*
     * auth_info = "Content-Encoding: auth" || 0x00
     * PRK_combine = HMAC-SHA-256(auth_secret, ecdh_secret)
     * IKM = HMAC-SHA-256(PRK_combine, auth_info || 0x01)
     * PRK = HMAC-SHA-256(salt, IKM)
     */
    static const auto authInfo = "Content-Encoding: auth\x00\x01"_span8;
    auto prkCombine = hmacSHA256(clientKeys.sharedAuthSecret, *ecdhSecretResult);
    auto ikm = hmacSHA256(prkCombine, authInfo);
    auto prk = hmacSHA256(salt, ikm);

    /*
     * context = "P-256" || 0x00 ||
     *           0x00 || 0x41 || ua_public ||
     *           0x00 || 0x41 || as_public
     *
     * Note that we also append a 0x01 byte at the end here since the cek and nonce
     * derivation functions below require that trailing 0x01 byte.
     */
    struct KeyDerivationContext {
        char label[6] = { "P-256" };
        std::array<uint8_t, 2> clientPublicKeyLength { 0, 0x41 };
        std::array<uint8_t, p256dhPublicKeyLength> clientPublicKey;
        std::array<uint8_t, 2> serverPublicKeyLength { 0, 0x41 };
        std::array<uint8_t, p256dhPublicKeyLength> serverPublicKey;
        uint8_t end = 0x01;
    };
    static_assert(sizeof(KeyDerivationContext) == 141);
    KeyDerivationContext context;
    memcpySpan(std::span { context.clientPublicKey }, clientKeys.clientP256DHKeyPair.publicKey.span().first(p256dhPublicKeyLength));
    memcpySpan(std::span { context.serverPublicKey }, serverP256DHPublicKey.first(p256dhPublicKeyLength));

    /*
     * cek_info = "Content-Encoding: aesgcm" || 0x00 || context
     * CEK = HMAC-SHA-256(PRK, cek_info || 0x01)[0..15]
     */
    static constexpr auto cekInfoHeader = "Content-Encoding: aesgcm"_s;
    std::array<uint8_t, cekInfoHeader.length() + 1 + sizeof(context)> cekInfo;
    memcpySpan(std::span { cekInfo }, cekInfoHeader.unsafeSpanIncludingNullTerminator());
    memcpySpan(std::span { cekInfo }.subspan(cekInfoHeader.length() + 1), asByteSpan(context));

    auto cek = hmacSHA256(prk, cekInfo);
    cek.shrink(16);

    /*
     * nonce_info = "Content-Encoding: nonce" || 0x00 || context
     * NONCE = HMAC-SHA-256(PRK, nonce_info || 0x01)[0..11]
     */
    static constexpr auto nonceInfoHeader = "Content-Encoding: nonce"_s;
    std::array<uint8_t, nonceInfoHeader.length() + 1 + sizeof(context)> nonceInfo;
    memcpySpan(std::span { nonceInfo }, nonceInfoHeader.unsafeSpanIncludingNullTerminator());
    memcpySpan(std::span { nonceInfo }.subspan(nonceInfoHeader.length() + 1), asByteSpan(context));

    auto nonce = hmacSHA256(prk, nonceInfo);
    nonce.shrink(12);

    // Finally, decrypt with AES128GCM and return the unpadded plaintext.
    auto plainTextResult = decryptAES128GCM(cek, nonce, payload);
    if (!plainTextResult)
        return std::nullopt;

    auto plainText = WTFMove(plainTextResult.value());
    size_t paddingLength = computeAESGCMPaddingLength(plainText.span());
    if (paddingLength == SIZE_MAX)
        return std::nullopt;

    return plainText.subvector(paddingLength);
}

} // namespace WebCore::PushCrypto
