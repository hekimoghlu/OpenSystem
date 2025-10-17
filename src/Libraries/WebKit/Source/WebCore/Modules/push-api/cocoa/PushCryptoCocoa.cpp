/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#include "PushCrypto.h"

#include <CommonCrypto/CommonHMAC.h>
#include <pal/spi/cocoa/CommonCryptoSPI.h>
#include <wtf/Scope.h>
#include <wtf/StdLibExtras.h>

namespace WebCore::PushCrypto {

P256DHKeyPair P256DHKeyPair::generate(void)
{
    CCECCryptorRef ccPublicKey = nullptr;
    CCECCryptorRef ccPrivateKey = nullptr;
    auto releaser = WTF::makeScopeExit([&ccPublicKey, &ccPrivateKey]() {
        if (ccPublicKey)
            CCECCryptorRelease(ccPublicKey);
        if (ccPrivateKey)
            CCECCryptorRelease(ccPrivateKey);
    });

    CCCryptorStatus status = CCECCryptorGeneratePair(256, &ccPublicKey, &ccPrivateKey);
    RELEASE_ASSERT(status == kCCSuccess);

    std::array<uint8_t, p256dhPublicKeyLength> publicKey;
    size_t publicKeyLength = publicKey.size();
    status = CCECCryptorExportKey(kCCImportKeyBinary, publicKey.data(), &publicKeyLength, ccECKeyPublic, ccPublicKey);
    RELEASE_ASSERT(status == kCCSuccess && publicKeyLength == p256dhPublicKeyLength);

    // CommonCrypto expects the binary format to be 65 byte public key followed by the 32 byte private key, so we want to extract the last 32 bytes from the buffer.
    std::array<uint8_t, p256dhPublicKeyLength + p256dhPrivateKeyLength> key;
    size_t keyLength = key.size();
    status = CCECCryptorExportKey(kCCImportKeyBinary, key.data(), &keyLength, ccECKeyPrivate, ccPrivateKey);
    RELEASE_ASSERT(status == kCCSuccess && keyLength == key.size());

    return P256DHKeyPair {
        Vector<uint8_t> { publicKey },
        Vector<uint8_t> { std::span { key }.subspan(p256dhPublicKeyLength, p256dhPrivateKeyLength) }
    };
}

bool validateP256DHPublicKey(std::span<const uint8_t> publicKey)
{
    CCECCryptorRef ccPublicKey = nullptr;
    CCCryptorStatus status = CCECCryptorImportKey(kCCImportKeyBinary, publicKey.data(), publicKey.size(), ccECKeyPublic, &ccPublicKey);
    if (!ccPublicKey)
        return false;
    CCECCryptorRelease(ccPublicKey);
    return status == kCCSuccess;
}

std::optional<Vector<uint8_t>> computeP256DHSharedSecret(std::span<const uint8_t> publicKey, const P256DHKeyPair& keyPair)
{
    if (publicKey.size() != p256dhPublicKeyLength || keyPair.publicKey.size() != p256dhPublicKeyLength || keyPair.privateKey.size() != p256dhPrivateKeyLength)
        return std::nullopt;

    CCECCryptorRef ccPublicKey = nullptr;
    CCECCryptorRef ccPrivateKey = nullptr;
    auto releaser = WTF::makeScopeExit([&ccPublicKey, &ccPrivateKey]() {
        if (ccPublicKey)
            CCECCryptorRelease(ccPublicKey);
        if (ccPrivateKey)
            CCECCryptorRelease(ccPrivateKey);
    });

    if (CCECCryptorImportKey(kCCImportKeyBinary, publicKey.data(), p256dhPublicKeyLength, ccECKeyPublic, &ccPublicKey) != kCCSuccess)
        return std::nullopt;

    // CommonCrypto expects the binary format to be 65 byte public key followed by the 32 byte private key.
    std::array<uint8_t, p256dhPublicKeyLength + p256dhPrivateKeyLength> keyBuffer;
    memcpySpan(std::span { keyBuffer }, keyPair.publicKey.span().first(p256dhPublicKeyLength));
    memcpySpan(std::span { keyBuffer }.subspan(p256dhPublicKeyLength), keyPair.privateKey.span().first(p256dhPrivateKeyLength));
    if (CCECCryptorImportKey(kCCImportKeyBinary, keyBuffer.data(), keyBuffer.size(), ccECKeyPrivate, &ccPrivateKey) != kCCSuccess)
        return std::nullopt;

    Vector<uint8_t> sharedSecret(p256dhSharedSecretLength);
    size_t sharedSecretLength = sharedSecret.size();
    if (CCECCryptorComputeSharedSecret(ccPrivateKey, ccPublicKey, sharedSecret.begin(), &sharedSecretLength) != kCCSuccess || sharedSecretLength != p256dhSharedSecretLength)
        return std::nullopt;

    return sharedSecret;
}

Vector<uint8_t> hmacSHA256(std::span<const uint8_t> key, std::span<const uint8_t> message)
{
    Vector<uint8_t> result(sha256DigestLength);
    CCHmac(kCCHmacAlgSHA256, key.data(), key.size(), message.data(), message.size(), result.begin());
    return result;
}

std::optional<Vector<uint8_t>> decryptAES128GCM(std::span<const uint8_t> key, std::span<const uint8_t> iv, std::span<const uint8_t> cipherTextWithTag)
{
    if (cipherTextWithTag.size() < aes128GCMTagLength)
        return std::nullopt;

    Vector<uint8_t> plainText(cipherTextWithTag.size() - aes128GCMTagLength);
    auto nonTagCipherTextLength = cipherTextWithTag.size() - aes128GCMTagLength;
    auto result = CCCryptorGCMOneshotDecrypt(kCCAlgorithmAES, key.data(), key.size(), iv.data(), iv.size(), nullptr /* additionalData */, 0 /* additionalDataLength */, cipherTextWithTag.data(), nonTagCipherTextLength, plainText.data(), cipherTextWithTag.subspan(nonTagCipherTextLength).data(), aes128GCMTagLength);
    if (result != kCCSuccess)
        return std::nullopt;

    return plainText;
}

} // namespace WebCore::PushCrypto
