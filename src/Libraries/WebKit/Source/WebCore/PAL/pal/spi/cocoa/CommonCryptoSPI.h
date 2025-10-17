/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include <CommonCrypto/CommonCrypto.h>

#if USE(APPLE_INTERNAL_SDK)

#include <CommonCrypto/CommonCryptorSPI.h>
#include <CommonCrypto/CommonDigestSPI.h>
#include <CommonCrypto/CommonECCryptor.h>
#include <CommonCrypto/CommonKeyDerivationSPI.h>
#include <CommonCrypto/CommonRSACryptor.h>
#include <CommonCrypto/CommonRSACryptorSPI.h>

#else

#ifndef _CC_RSACRYPTOR_H_
enum {
    kCCDigestNone = 0,
    kCCDigestSHA1 = 8,
    kCCDigestSHA224 = 9,
    kCCDigestSHA256 = 10,
    kCCDigestSHA384 = 11,
    kCCDigestSHA512 = 12,
};
typedef uint32_t CCDigestAlgorithm;

enum {
    ccRSAKeyPublic = 0,
    ccRSAKeyPrivate = 1
};
typedef uint32_t CCRSAKeyType;

enum {
    ccPKCS1Padding = 1001,
    ccOAEPPadding = 1002,
    ccRSAPSSPadding = 1005
};
typedef uint32_t CCAsymmetricPadding;

typedef struct CCKDFParameters *CCKDFParametersRef;

#endif

typedef struct _CCRSACryptor *CCRSACryptorRef;
extern "C" CCCryptorStatus CCRSACryptorEncrypt(CCRSACryptorRef publicKey, CCAsymmetricPadding padding, const void *plainText, size_t plainTextLen, void *cipherText, size_t *cipherTextLen, const void *tagData, size_t tagDataLen, CCDigestAlgorithm digestType);
extern "C" CCCryptorStatus CCRSACryptorDecrypt(CCRSACryptorRef privateKey, CCAsymmetricPadding padding, const void *cipherText, size_t cipherTextLen, void *plainText, size_t *plainTextLen, const void *tagData, size_t tagDataLen, CCDigestAlgorithm digestType);
extern "C" CCCryptorStatus CCRSACryptorSign(CCRSACryptorRef privateKey, CCAsymmetricPadding padding, const void *hashToSign, size_t hashSignLen, CCDigestAlgorithm digestType, size_t saltLen, void *signedData, size_t *signedDataLen);
extern "C" CCCryptorStatus CCRSACryptorVerify(CCRSACryptorRef publicKey, CCAsymmetricPadding padding, const void *hash, size_t hashLen, CCDigestAlgorithm digestType, size_t saltLen, const void *signedData, size_t signedDataLen);
extern "C" CCCryptorStatus CCRSACryptorGeneratePair(size_t keysize, uint32_t e, CCRSACryptorRef *publicKey, CCRSACryptorRef *privateKey);
extern "C" CCRSACryptorRef CCRSACryptorGetPublicKeyFromPrivateKey(CCRSACryptorRef privkey);
extern "C" void CCRSACryptorRelease(CCRSACryptorRef key);
extern "C" CCCryptorStatus CCRSAGetKeyComponents(CCRSACryptorRef rsaKey, uint8_t *modulus, size_t *modulusLength, uint8_t *exponent, size_t *exponentLength, uint8_t *p, size_t *pLength, uint8_t *q, size_t *qLength);
extern "C" CCRSAKeyType CCRSAGetKeyType(CCRSACryptorRef key);
extern "C" CCCryptorStatus CCRSACryptorImport(const void *keyPackage, size_t keyPackageLen, CCRSACryptorRef *key);
extern "C" CCCryptorStatus CCRSACryptorExport(CCRSACryptorRef key, void *out, size_t *outLen);

extern "C" CCCryptorStatus CCRSAGetCRTComponentsSizes(CCRSACryptorRef rsaKey, size_t *dpSize, size_t *dqSize, size_t *qinvSize);
extern "C" CCCryptorStatus CCRSAGetCRTComponents(CCRSACryptorRef rsaKey, void *dp, size_t dpSize, void *dq, size_t dqSize, void *qinv, size_t qinvSize);

#ifndef _CC_ECCRYPTOR_H_
enum {
    ccECKeyPublic = 0,
    ccECKeyPrivate = 1,
};
typedef uint32_t CCECKeyType;

enum {
    kCCImportKeyBinary = 0,
    kCCImportKeyCompact = 2,
};
typedef uint32_t CCECKeyExternalFormat;
#endif

typedef struct _CCECCryptor *CCECCryptorRef;
extern "C" CCCryptorStatus CCECCryptorGeneratePair(size_t keysize, CCECCryptorRef *publicKey, CCECCryptorRef *privateKey);
extern "C" void CCECCryptorRelease(CCECCryptorRef key);
extern "C" CCCryptorStatus CCECCryptorImportKey(CCECKeyExternalFormat format, const void *keyPackage, size_t keyPackageLen, CCECKeyType keyType, CCECCryptorRef *key);
extern "C" CCCryptorStatus CCECCryptorExportKey(CCECKeyExternalFormat format, void *keyPackage, size_t *keyPackageLen, CCECKeyType keyType, CCECCryptorRef key);
extern "C" int CCECGetKeySize(CCECCryptorRef key);
extern "C" CCCryptorStatus CCECCryptorCreateFromData(size_t keySize, uint8_t *qX, size_t qXLength, uint8_t *qY, size_t qYLength, CCECCryptorRef *ref);
extern "C" CCCryptorStatus CCECCryptorGetKeyComponents(CCECCryptorRef ecKey, size_t *keySize, uint8_t *qX, size_t *qXLength, uint8_t *qY, size_t *qYLength, uint8_t *d, size_t *dLength);
extern "C" CCCryptorStatus CCECCryptorComputeSharedSecret(CCECCryptorRef privateKey, CCECCryptorRef publicKey, void *out, size_t *outLen);
extern "C" CCCryptorStatus CCECCryptorSignHash(CCECCryptorRef privateKey, const void *hashToSign, size_t hashSignLen, void *signedData, size_t *signedDataLen);
extern "C" CCCryptorStatus CCECCryptorVerifyHash(CCECCryptorRef publicKey, const void *hash, size_t hashLen, const void *signedData, size_t signedDataLen, uint32_t *valid);

#ifndef CommonCrypto_CommonNistKeyDerivation_h
enum {
    kCCKDFAlgorithmHKDF = 6
};
typedef uint32_t CCKDFAlgorithm;
#endif

extern "C" CCStatus CCKeyDerivationHMac(CCKDFAlgorithm algorithm, CCDigestAlgorithm digest, unsigned rounds, const void *keyDerivationKey, size_t keyDerivationKeyLen, const void *label, size_t labelLen, const void *context, size_t contextLen, const void *iv, size_t ivLen, const void *salt, size_t saltLen, void *derivedKey, size_t derivedKeyLen);

extern "C" CCStatus CCDeriveKey(const CCKDFParametersRef params, CCDigestAlgorithm digest, const void *keyDerivationKey, size_t keyDerivationKeyLen, void *derivedKey, size_t derivedKeyLen);
extern "C" CCStatus CCKDFParametersCreateHkdf(CCKDFParametersRef *params, const void *salt, size_t saltLen, const void *context, size_t contextLen);
extern "C" void CCKDFParametersDestroy(CCKDFParametersRef params);
extern "C" CCCryptorStatus CCCryptorGCM(CCOperation op, CCAlgorithm alg, const void* key, size_t keyLength, const void* iv, size_t ivLen, const void* aData, size_t aDataLen, const void* dataIn, size_t dataInLength, void* dataOut, void* tag, size_t* tagLength);
extern "C" CCCryptorStatus CCCryptorGCMOneshotDecrypt(CCAlgorithm alg, const void *key, size_t keyLength, const void  *iv, size_t ivLen, const void  *aData, size_t aDataLen, const void *dataIn, size_t dataInLength, void        *dataOut, const void  *tagIn, size_t tagLength);
extern "C" CCCryptorStatus CCRSACryptorCreateFromData(CCRSAKeyType keyType, const uint8_t *modulus, size_t modulusLength, const uint8_t *exponent, size_t exponentLength, const uint8_t *p, size_t pLength, const uint8_t *q, size_t qLength, CCRSACryptorRef *ref);

#endif // !USE(APPLE_INTERNAL_SDK)
