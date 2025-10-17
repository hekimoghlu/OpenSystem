/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#import <XCTest/XCTest.h>
#import <Security/SecKeyPriv.h>
#import <Security/SecItemPriv.h>
#include <corecrypto/ccec.h>
#include <corecrypto/ccec448.h>
#include <corecrypto/ccec448_priv.h>
#include <corecrypto/ccsha2.h>
#include <libDER/oids.h>


@interface SecKeyCurve448Tests : XCTestCase
@end

@implementation SecKeyCurve448Tests

- (void)verifySignFunctionalityWithPrivateKey:(SecKeyRef)privatekey
                                    publicKey:(SecKeyRef)pubkey // optional
                             invalidPublicKey:(SecKeyRef)invalidpubkey // optional
{
    XCTAssert(privatekey, @"no private key provided");
    NSData *message = [@"message" dataUsingEncoding:NSUTF8StringEncoding];
    NSData *signature = nil;
    NSError *error = nil;
    SecKeyAlgorithm algorithm;
    int err = -1;

    // sign data with invalid aglorithm
    algorithm = kSecKeyAlgorithmECDSASignatureMessageX962SHA512;
    signature = CFBridgingRelease(SecKeyCreateSignature(privatekey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNil(signature, @"Signing succeeded unexpectadly");

    // sign data
    algorithm = kSecKeyAlgorithmEdDSASignatureMessageCurve448SHAKE256 ;
    signature = CFBridgingRelease(SecKeyCreateSignature(privatekey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNotNil(signature, @"Signing failed: %@", error);
    XCTAssert(signature.length == sizeof(cced448signature), @"Invalid signature length: %u", (unsigned)signature.length);

    // copy out public key
    id localPubKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privatekey));
    XCTAssert(localPubKey, @"Failed to export pubkey");
    NSData *pubKeyData = CFBridgingRelease(SecKeyCopyExternalRepresentation((SecKeyRef)localPubKey, (void *)&error));
    XCTAssert(pubKeyData && pubKeyData.length == sizeof(cced448pubkey), @"Failed to export pubkey data: %@", error);
    
    // verify the signature with CoreCrypto
    if (signature) {
        err = cced448_verify(message.length, message.bytes, signature.bytes, pubKeyData.bytes);
        XCTAssert(err == 0, @"Signature verification failed with CoreCrypto, err: %d", err);
    }
    
    // verify the signature with SecKey
    Boolean valid = SecKeyVerifySignature((SecKeyRef)localPubKey, algorithm, (CFDataRef)message, (CFDataRef)signature, (void*)&error);
    XCTAssertTrue(valid, @"Signature verification failed with SecKey");
    
    NSMutableData *invalidSignature = [signature mutableCopy];
    ((uint8_t*)invalidSignature.mutableBytes)[0] += 1; // make it invalid
    
    // verify the invalid signature with CoreCrypto
    err = cced448_verify(message.length, message.bytes, invalidSignature.bytes, pubKeyData.bytes);
    XCTAssert(err != 0, @"Signature verification succeeded unexpectedly with CoreCrypto, err: %d", err);
    
    // verify the signature with SecKey
    valid = SecKeyVerifySignature((SecKeyRef)localPubKey, algorithm, (CFDataRef)message, (CFDataRef)invalidSignature, (void*)&error);
    XCTAssertFalse(valid, @"Signature verification succeeded unexpectadly with SecKey");
    
    if (pubkey) {
        // verify the signature with the testing key
        valid = SecKeyVerifySignature(pubkey, algorithm, (CFDataRef)message, (CFDataRef)signature, (void*)&error);
        XCTAssertTrue(valid, @"Signature verification failed with testing SecKey");
    }
    if (invalidpubkey) {
        // verify the signature with invalid testing key
        valid = SecKeyVerifySignature(invalidpubkey, algorithm, (CFDataRef)message, (CFDataRef)signature, (void*)&error);
        XCTAssertFalse(valid, @"Signature verification succeeded unexpectadly with SecKey");
    }
}

-(void)verifyEncryptionFunctionalityWithKey:(id)privateKey algorithm:(SecKeyAlgorithm)algorithm {
    NSError *error = nil;
    id publicKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privateKey));
    XCTAssert(publicKey, "public key retrieved");

    XCTAssert(SecKeyIsAlgorithmSupported((SecKeyRef)privateKey, kSecKeyOperationTypeDecrypt, algorithm),
       "%@ supported for decryption", algorithm);
    XCTAssert(SecKeyIsAlgorithmSupported((SecKeyRef)publicKey, kSecKeyOperationTypeEncrypt, algorithm),
       "%@ supported for encryption", algorithm);
    XCTAssert(!SecKeyIsAlgorithmSupported((SecKeyRef)publicKey, kSecKeyOperationTypeDecrypt, algorithm),
       "%@ not supported for decryption - pubkey", algorithm);
    XCTAssert(!SecKeyIsAlgorithmSupported((SecKeyRef)privateKey, kSecKeyOperationTypeEncrypt, algorithm),
       "%@ not supported for encryption - privKey", algorithm);

    NSData *message = [NSData dataWithBytes:"hello" length:5];
    NSDictionary *sharedInfo = @{(id)kSecKeyKeyExchangeParameterSharedInfo :[NSData dataWithBytes:"shared" length:6]};
    error = nil;
    NSData *encrypted = CFBridgingRelease(SecKeyCreateEncryptedData((SecKeyRef)publicKey, algorithm, (CFDataRef)message, (void *)&error));
    NSData *encryptedSI = CFBridgingRelease(SecKeyCreateEncryptedDataWithParameters((__bridge SecKeyRef)publicKey, algorithm, (__bridge CFDataRef)message, (__bridge CFDictionaryRef)sharedInfo, (void *)&error));
    XCTAssert(encrypted, "message encrypted");
    XCTAssert(encryptedSI, "message encrypted");

    error = nil;
    NSData *decrypted = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privateKey, algorithm, (CFDataRef)encrypted, (void *)&error));
    XCTAssert(decrypted, "encrypted message decrypted");
    XCTAssert([decrypted isEqual:message], "decrypted message is equal as original one (original:%@ decrypted:%@)", message, decrypted);
    NSData *decryptedSI = CFBridgingRelease(SecKeyCreateDecryptedDataWithParameters((__bridge SecKeyRef)privateKey, algorithm, (__bridge CFDataRef)encryptedSI, (__bridge CFDictionaryRef)sharedInfo, (void *)&error));
    XCTAssert(decryptedSI, "encrypted-with-sharedinfo message decrypted: %@", error);
    XCTAssert([decryptedSI isEqual:message], "decrypted-with-sharedinfo message is equal as original one (original:%@ decrypted:%@)", message, decryptedSI);

    // Modify encrypted message and verify that it cannot be decrypted.
    NSMutableData *badEncrypted = [NSMutableData dataWithData:encrypted];
    UInt8 *badEncryptedBuffer = badEncrypted.mutableBytes;
    badEncryptedBuffer[badEncrypted.length - 8] ^= 0xff;

    error = nil;
    decrypted = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privateKey, algorithm, (CFDataRef)badEncrypted, (void *)&error));
    XCTAssert(decrypted == nil, "broken encrypted message failed to decrypt (tag breakage)");

    badEncrypted = [NSMutableData dataWithData:encrypted];
    badEncryptedBuffer = badEncrypted.mutableBytes;
    badEncryptedBuffer[badEncrypted.length - 20] ^= 0xff;

    error = nil;
    decrypted = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privateKey, algorithm, (CFDataRef)badEncrypted, (void *)&error));
    XCTAssert(decrypted == nil, "broken encrypted message failed to decrypt (ciphertext breakage)");
    
    badEncrypted = [NSMutableData dataWithData:encrypted];
    badEncryptedBuffer = badEncrypted.mutableBytes;
    badEncryptedBuffer[0] ^= 0xff;

    error = nil;
    decrypted = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privateKey, algorithm, (CFDataRef)badEncrypted, (void *)&error));
    XCTAssert(decrypted == nil, "broken encrypted message failed to decrypt (pubkey intro breakage)");

    badEncrypted = [NSMutableData dataWithData:encrypted];
    badEncryptedBuffer = badEncrypted.mutableBytes;
    badEncryptedBuffer[1] ^= 0xff;

    error = nil;
    decrypted = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privateKey, algorithm, (CFDataRef)badEncrypted, (void *)&error));
    XCTAssert(decrypted == nil, "broken encrypted message failed to decrypt (pubkey data breakage)");
}

- (void)testSignEd448 {
    NSError *error = nil;
    
    // generate private key
    id privateKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeySizeInBits: @256, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssertNotNil(privateKey, @"key generation failed: %@", error);
    
    // verify signing functionality
    [self verifySignFunctionalityWithPrivateKey:(SecKeyRef)privateKey publicKey:NULL invalidPublicKey:NULL];
}

- (void)testExportImportEd448PublicKey {
    NSError *error = nil;

    // generate private key
    id privateKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeySizeInBits: @256, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssertNotNil(privateKey, @"key generation failed: %@", error);
    
    // copy out public key
    id pubKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privateKey));
    XCTAssert(pubKey, @"Failed to export pubkey");
    NSData *pubKeyData = CFBridgingRelease(SecKeyCopyExternalRepresentation((SecKeyRef)pubKey, (void *)&error));
    XCTAssert(pubKeyData && pubKeyData.length == sizeof(cced448pubkey), @"Failed to export pubkey data: %@", error);

    // import public key
    id importedPubKey = CFBridgingRelease(SecKeyCreateWithData((CFDataRef)pubKeyData, (CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPublic, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssert(importedPubKey, @"Failed to import public key: %@", error);
    
    // import invalid public key
    NSMutableData *invalidPubKeyData = [pubKeyData mutableCopy];
    ((uint8_t*)invalidPubKeyData.mutableBytes)[0] += 1; // make it invalid

    id importedInvalidPubKey = CFBridgingRelease(SecKeyCreateWithData((CFDataRef)invalidPubKeyData, (CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPublic, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssert(importedInvalidPubKey, @"Failed to import public key: %@", error);

    // verify signing functionality
    [self verifySignFunctionalityWithPrivateKey:(SecKeyRef)privateKey
                                      publicKey:(SecKeyRef)importedPubKey
                               invalidPublicKey:(SecKeyRef)importedInvalidPubKey];
}

- (void)testExportImportEd448rivateKey {
    NSError *error = nil;

    // generate private key
    id privateKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeySizeInBits: @256, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssertNotNil(privateKey, @"key generation failed: %@", error);

    // copy out public key
    id pubKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privateKey));
    XCTAssert(pubKey, @"Failed to export pubkey");

    // copy out private key
    NSData *privateKeyData = CFBridgingRelease(SecKeyCopyExternalRepresentation((SecKeyRef)privateKey, (void *)&error));
    XCTAssert(privateKeyData && privateKeyData.length == sizeof(cced448secretkey), @"Failed to export private key data: %@", error);
    
    // import private key
    id importedPrivateKey = CFBridgingRelease(SecKeyCreateWithData((CFDataRef)privateKeyData, (CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssert(importedPrivateKey, @"Failed to import private key: %@", error);
    
    // verify signing functionality with imported private key and original public key
    [self verifySignFunctionalityWithPrivateKey:(SecKeyRef)importedPrivateKey
                                      publicKey:(SecKeyRef)pubKey
                               invalidPublicKey:NULL];
}

- (void)testEncryptX448 {
    // generate private key
    NSError *error = nil;
    id privateKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeX448, (id)kSecAttrKeySizeInBits: @256, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssertNotNil(privateKey, @"key generation failed: %@", error);

    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardX963SHA1AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardX963SHA224AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardX963SHA256AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardX963SHA384AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardX963SHA512AESGCM];
    
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA224AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA256AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA384AESGCM];
    [self verifyEncryptionFunctionalityWithKey:privateKey algorithm:kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA512AESGCM];
}

- (void)testSingEd448CoreCrypto {
    NSData *message = [@"message" dataUsingEncoding:NSUTF8StringEncoding];

    struct ccrng_state *rng = ccrng(NULL);

    cced448secretkey privatekey = {};
    cced448pubkey pubkeyTemp = {};

    int err = cced448_make_key_pair(rng, pubkeyTemp, privatekey);
    XCTAssertTrue(err == 0, @"Failed to generate keypair using CoreCrypto, err=%d", err);

    cced448pubkey pubkey = {};
    err = cced448_make_pub(rng, pubkey, privatekey);
    XCTAssertTrue(err == 0, @"Failed to generate public key from private using CoreCrypto, err=%d", err);

    cced448signature signature = {};
    const uint8_t *msg = message.bytes;
    size_t msgLen = message.length;
    
    err = cced448_sign(rng, signature, msgLen, msg, pubkey, privatekey);
    XCTAssertTrue(err == 0, @"Failed to sign using CoreCrypto, err=%d", err);
}

- (void)testDHX448CoreCrypto {
    struct ccrng_state *rng = ccrng(NULL);

    ccec448secretkey privateKey = {};
    ccec448pubkey pubKey = {};
    int err = cccurve448_make_key_pair(rng, pubKey, privateKey);
    XCTAssertTrue(err == 0, @"Failed to generate keypair using CoreCrypto, err=%d", err);

    ccec448secretkey ephemeralPrivateKey = {};
    ccec448pubkey ephemeralPubKey = {};
    err = cccurve448_make_key_pair(rng, ephemeralPubKey, ephemeralPrivateKey);
    XCTAssertTrue(err == 0, @"Failed to generate keypair using CoreCrypto, err=%d", err);

    ccec448key sh1;
    err = cccurve448(rng, sh1, ephemeralPrivateKey, pubKey);
    XCTAssertTrue(err == 0, @"Failed to perfrom DH using CoreCrypto, err=%d", err);

    ccec448key sh2;
    err = cccurve448(rng, sh2, privateKey, ephemeralPubKey);
    XCTAssertTrue(err == 0, @"Failed to perfrom DH using CoreCrypto, err=%d", err);

    XCTAssert(0 == memcmp(sh1, sh2, sizeof(sh1)), @"DH secrets do not match");
}

- (void)testSPKIForEd448 {
    NSError *error = nil;
    id privateKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)@{(id)kSecAttrKeyType: (id)kSecAttrKeyTypeEd448, (id)kSecAttrKeySizeInBits: @256, (id)kSecUseDataProtectionKeychain : @YES}, (void *)&error));
    XCTAssertNotNil(privateKey, @"key generation failed: %@", error);

    id publicKey = CFBridgingRelease(SecKeyCopyPublicKey((__bridge SecKeyRef)privateKey));
    XCTAssertNotNil(publicKey, @"public key computation failed");

    NSData *spki = CFBridgingRelease(SecKeyCopySubjectPublicKeyInfo((__bridge SecKeyRef)publicKey));
    XCTAssertNotNil(spki, @"failed to create spki from public key");

    id decodedKey = CFBridgingRelease(SecKeyCreateFromSubjectPublicKeyInfoData(NULL, (__bridge CFDataRef)spki));
    XCTAssertNotNil(decodedKey, @"failed to create key from spki");
    XCTAssertEqualObjects(publicKey, decodedKey, @"failed to get same public key back from spki");
}

@end
