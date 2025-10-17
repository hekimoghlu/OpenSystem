/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "VirtualAuthenticatorUtils.h"

#if ENABLE(WEB_AUTHN)

#include <WebCore/WebAuthenticationConstants.h>
#include <WebCore/WebAuthenticationUtils.h>
#include <pal/crypto/CryptoDigest.h>
#include <wtf/cocoa/TypeCastsCocoa.h>
#include <wtf/cocoa/VectorCocoa.h>

namespace WebKit {
using namespace WebCore;

uint8_t flagsForConfig(const VirtualAuthenticatorConfiguration& config)
{
    uint8_t flags = WebAuthn::attestedCredentialDataIncludedFlag;
    if (config.isUserConsenting)
        flags = flags | WebAuthn::userPresenceFlag;
    if (config.isUserVerified)
        flags = flags | WebAuthn::userVerifiedFlag;
    return flags;
}

RetainPtr<SecKeyRef> createPrivateKey()
{
    NSDictionary* options = @{
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate,
        (id)kSecAttrKeySizeInBits: @256,
    };
    CFErrorRef errorRef = nullptr;
    auto key = adoptCF(SecKeyCreateRandomKey(
        (__bridge CFDictionaryRef)options,
        &errorRef
    ));
    if (errorRef)
        return nullptr;
    return key;
}

std::pair<Vector<uint8_t>, Vector<uint8_t>> credentialIdAndCosePubKeyForPrivateKey(RetainPtr<SecKeyRef> privateKey)
{
    RetainPtr<CFDataRef> publicKeyDataRef;
    {
        auto publicKey = adoptCF(SecKeyCopyPublicKey(privateKey.get()));
        CFErrorRef errorRef = nullptr;
        publicKeyDataRef = adoptCF(SecKeyCopyExternalRepresentation(publicKey.get(), &errorRef));
        auto retainError = adoptCF(errorRef);
        ASSERT(!errorRef);
        ASSERT(((NSData *)publicKeyDataRef.get()).length == (1 + 2 * ES256FieldElementLength)); // 04 | X | Y
    }
    NSData *nsPublicKeyData = (NSData *)publicKeyDataRef.get();

    Vector<uint8_t> credentialId;
    {
        auto digest = PAL::CryptoDigest::create(PAL::CryptoDigest::Algorithm::SHA_1);
        digest->addBytes(span(nsPublicKeyData));
        credentialId = digest->computeHash();
    }

    Vector<uint8_t> cosePublicKey;
    {
        // COSE Encoding
        Vector<uint8_t> x(ES256FieldElementLength);
        [nsPublicKeyData getBytes: x.data() range:NSMakeRange(1, ES256FieldElementLength)];
        Vector<uint8_t> y(ES256FieldElementLength);
        [nsPublicKeyData getBytes: y.data() range:NSMakeRange(1 + ES256FieldElementLength, ES256FieldElementLength)];
        cosePublicKey = encodeES256PublicKeyAsCBOR(WTFMove(x), WTFMove(y));
    }
    return std::pair { credentialId, cosePublicKey };
}

String base64PrivateKey(RetainPtr<SecKeyRef> privateKey)
{
    CFErrorRef errorRef = nullptr;
    auto privateKeyRep = adoptCF(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)((id)privateKey.get()), &errorRef));
    auto retainError = adoptCF(errorRef);
    if (errorRef) {
        ASSERT_NOT_REACHED();
        return emptyString();
    }
    NSData *nsPrivateKeyRep = (NSData *)privateKeyRep.get();

    return String([nsPrivateKeyRep base64EncodedStringWithOptions:0]);
}

RetainPtr<SecKeyRef> privateKeyFromBase64(const String& base64PrivateKey)
{
    NSDictionary* options = @{
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate,
        (id)kSecAttrKeySizeInBits: @256,
    };
    RetainPtr<NSData> privateKey = adoptNS([[NSData alloc] initWithBase64EncodedString:base64PrivateKey options:0]);
    CFErrorRef errorRef = nullptr;
    auto key = adoptCF(SecKeyCreateWithData(
        bridge_cast(privateKey.get()),
        bridge_cast(options),
        &errorRef
    ));
    ASSERT(!errorRef);
    return key;
}

Vector<uint8_t> signatureForPrivateKey(RetainPtr<SecKeyRef> privateKey, const Vector<uint8_t>& authData, const Vector<uint8_t>& clientDataHash)
{
    NSMutableData *dataToSign = [NSMutableData dataWithBytes:authData.data() length:authData.size()];
    [dataToSign appendBytes:clientDataHash.data() length:clientDataHash.size()];
    RetainPtr<CFDataRef> signature;
    {
        CFErrorRef errorRef = nullptr;
        signature = adoptCF(SecKeyCreateSignature((__bridge SecKeyRef)((id)privateKey.get()), kSecKeyAlgorithmECDSASignatureMessageX962SHA256, (__bridge CFDataRef)dataToSign, &errorRef));
        auto retainError = adoptCF(errorRef);
        ASSERT(!errorRef);
    }

    return makeVector((NSData *)signature.get());
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
