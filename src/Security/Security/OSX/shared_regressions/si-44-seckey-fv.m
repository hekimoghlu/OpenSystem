/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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

//
//  si-44-seckey-fv.m
//

#import <Foundation/Foundation.h>

#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
#import "SecureKeyVaultPublic.h"
#import <Security/SecKey.h>

#import "shared_regressions.h"

static void testFileVaultKeyRawSign(void) {
    id key = CFBridgingRelease(SecKeyCreateWithSecureKeyVaultID(kCFAllocatorDefault, kSecureKeyVaultIAPAuthPrivateKey));
    id certificate = CFBridgingRelease(SecCertificateCreateWithSecureKeyVaultID(kCFAllocatorDefault, kSecureKeyVaultIAPAuthPrivateKey));
    id pubKey = CFBridgingRelease(SecCertificateCopyKey((SecCertificateRef)certificate));

    uint8_t hash[20] = { 0 };
    uint8_t signature[256] = { 0 };
    size_t siglen = sizeof(signature);
    ok_status(SecKeyRawSign((SecKeyRef)key, kSecPaddingPKCS1SHA1, hash, sizeof(hash), signature, &siglen), "rawSign for fileVault failed");
    ok_status(SecKeyRawVerify((SecKeyRef)pubKey, kSecPaddingPKCS1SHA1, hash, sizeof(hash), signature, siglen), "rawverify for fileVault failed");
}

static void testFileVaultKeySign(void) {
    NSData *data = [@"dataToSign" dataUsingEncoding:NSUTF8StringEncoding];
    NSData *signature;
    SecKeyAlgorithm algorithm = NULL;
    NSError *error;
    id key = CFBridgingRelease(SecKeyCreateWithSecureKeyVaultID(kCFAllocatorDefault, kSecureKeyVaultIAPAuthPrivateKey));
    id certificate = CFBridgingRelease(SecCertificateCreateWithSecureKeyVaultID(kCFAllocatorDefault, kSecureKeyVaultIAPAuthPrivateKey));
    id pubKey = CFBridgingRelease(SecCertificateCopyKey((SecCertificateRef)certificate));

    algorithm = kSecKeyAlgorithmRSASignatureMessagePKCS1v15SHA1;
    error = nil;
    signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)key, algorithm, (CFDataRef)data, (void *)&error));
    ok(signature != NULL, "signing with alg %@ failed, err %@", algorithm, error);
    ok(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)data, (CFDataRef)signature, (void *)&error));

    algorithm = kSecKeyAlgorithmRSASignatureMessagePKCS1v15SHA256;
    error = nil;
    signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)key, algorithm, (CFDataRef)data, (void *)&error));
    ok(signature != NULL, "signing with alg %@ failed, err %@", algorithm, error);
    ok(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)data, (CFDataRef)signature, (void *)&error));

    algorithm = kSecKeyAlgorithmRSASignatureMessagePSSSHA1;
    error = nil;
    signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)key, algorithm, (CFDataRef)data, (void *)&error));
    ok(signature != NULL, "signing with alg %@ failed, err %@", algorithm, error);
    ok(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)data, (CFDataRef)signature, (void *)&error));

    algorithm = kSecKeyAlgorithmRSASignatureMessagePSSSHA256;
    error = nil;
    signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)key, algorithm, (CFDataRef)data, (void *)&error));
    ok(signature != NULL, "signing with alg %@ failed, err %@", algorithm, error);
    ok(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)data, (CFDataRef)signature, (void *)&error));
}

int si_44_seckey_fv(int argc, char *const *argv) {
    @autoreleasepool {
        plan_tests(10);
        testFileVaultKeyRawSign();
        testFileVaultKeySign();
        return 0;
    }
}

#endif
