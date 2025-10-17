/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#import <Security/SecItemPriv.h>
#import <LocalAuthentication/LocalAuthentication.h>

@interface CTKIntegrationTests : XCTestCase

@end

@implementation CTKIntegrationTests

- (void)testItemAddQueryDelete {
    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
            (id)kSecReturnRef: @YES
        };
        id result;
        XCTAssertEqual(SecItemAdd((CFDictionaryRef)query, (void *)&result), errSecSuccess, @"Failed to generate key");
        XCTAssertEqual(CFGetTypeID((__bridge CFTypeRef)result), SecKeyGetTypeID(), @"Expected SecKey, got %@", result);
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecReturnRef: @YES
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecSuccess, @"ItemCopyMatching failed");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore
        };
        OSStatus status = SecItemDelete((CFDictionaryRef)query);
        XCTAssertEqual(status, errSecSuccess, @"Deletion failed");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecReturnRef: @YES
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecItemNotFound, @"ItemCopyMatching should not find deleted item");
    }
}

#if TARGET_OS_OSX // not yet for embedded
- (void)testSystemKeychainItemAddQueryDelete {
    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
            (id)kSecUseSystemKeychainAlways: @YES,
            (id)kSecReturnRef: @YES
        };
        id result;
        XCTAssertEqual(SecItemAdd((CFDictionaryRef)query, (void *)&result), errSecSuccess, @"Failed to generate key");
        XCTAssertEqual(CFGetTypeID((__bridge CFTypeRef)result), SecKeyGetTypeID(), @"Expected SecKey, got %@", result);
        NSDictionary *attributes = CFBridgingRelease(SecKeyCopyAttributes((SecKeyRef)result));
        XCTAssertNotNil(attributes);
        XCTAssertEqualObjects(attributes[(id)kSecUseSystemKeychainAlways], @YES);
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecUseSystemKeychainAlways: @YES,
            (id)kSecReturnRef: @YES,
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecSuccess, @"ItemCopyMatching failed");
        NSDictionary *attributes = CFBridgingRelease(SecKeyCopyAttributes((SecKeyRef)result));
        XCTAssertNotNil(attributes);
        XCTAssertEqualObjects(attributes[(id)kSecUseSystemKeychainAlways], @YES);
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecReturnRef: @YES,
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecItemNotFound, @"ItemCopyMatching should not find item in non-system keychain");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecUseSystemKeychainAlways: @YES,
        };
        OSStatus status = SecItemDelete((CFDictionaryRef)query);
        XCTAssertEqual(status, errSecSuccess, @"Deletion failed");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecUseSystemKeychainAlways: @YES,
            (id)kSecReturnRef: @YES,
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecItemNotFound, @"ItemCopyMatching should not find deleted item");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecReturnRef: @YES,
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecItemNotFound, @"ItemCopyMatching should not find item in non-system keychain");
    }
}
#endif

- (void)testProtectedItemsAddQueryDelete {
    NSData *password = [@"password" dataUsingEncoding:NSUTF8StringEncoding];
    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleWhenUnlocked, kSecAccessControlApplicationPassword | kSecAccessControlPrivateKeyUsage, (void *)&error));
        XCTAssertNotNil(sac);
        LAContext *authContext = [[LAContext alloc] init];
        [authContext setCredential:password type:LACredentialTypeApplicationPassword];

        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
            (id)kSecAttrAccessControl: sac,
            (id)kSecUseAuthenticationContext: authContext,
            (id)kSecReturnRef: @YES
        };
        id result;
        XCTAssertEqual(SecItemAdd((CFDictionaryRef)query, (void *)&result), errSecSuccess, @"Failed to generate key");
        XCTAssertEqual(CFGetTypeID((__bridge CFTypeRef)result), SecKeyGetTypeID(), @"Expected SecKey, got %@", result);
    }

    @autoreleasepool {
        LAContext *authContext = [[LAContext alloc] init];
        [authContext setCredential:password type:LACredentialTypeApplicationPassword];
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecUseAuthenticationContext: authContext,
            (id)kSecReturnRef: @YES
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecSuccess, @"ItemCopyMatching failed");
    }

    @autoreleasepool {
        LAContext *authContext = [[LAContext alloc] init];
        [authContext setCredential:password type:LACredentialTypeApplicationPassword];
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecUseAuthenticationContext: authContext
        };
        OSStatus status = SecItemDelete((CFDictionaryRef)query);
        XCTAssertEqual(status, errSecSuccess, @"Deletion failed");
    }

    @autoreleasepool {
        NSDictionary *query = @{
            (id)kSecClass: (id)kSecClassKey,
            (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
            (id)kSecReturnRef: @YES
        };
        id result;
        OSStatus status = SecItemCopyMatching((CFDictionaryRef)query, (void *)&result);
        XCTAssertEqual(status, errSecItemNotFound, @"ItemCopyMatching should not find deleted item");
    }
}

- (void)testSecKeyOperations {
    NSError *error;
    NSDictionary *attributes = @{
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
        (id)kSecAttrIsPermanent: @YES,
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom
    };
    id privKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)attributes, (void *)&error));
    XCTAssertNotNil(privKey, @"Failed to generate key, error %@", error);

    // Get key attributes
    attributes = CFBridgingRelease(SecKeyCopyAttributes((SecKeyRef)privKey));
    XCTAssertNotNil(attributes, @"Failed to get key attributes");
    XCTAssertEqualObjects(attributes[(id)kSecAttrKeyClass], (id)kSecAttrKeyClassPrivate);
    XCTAssertEqualObjects(attributes[(id)kSecAttrTokenID], (id)kSecAttrTokenIDAppleKeyStore);
    XCTAssertEqualObjects(attributes[(id)kSecAttrKeyType], (id)kSecAttrKeyTypeECSECPrimeRandom);
    XCTAssertEqualObjects(attributes[(id)kSecAttrKeySizeInBits], @256);

    // Get key attributes through keychain
    NSDictionary *query = @{
        (id)kSecClass: (id)kSecClassKey,
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
        (id)kSecReturnAttributes: @YES
    };
    attributes = nil;
    XCTAssertEqual(SecItemCopyMatching((CFDictionaryRef)query, (void *)&attributes), errSecSuccess);
    XCTAssertNotNil(attributes, @"Failed to get key attributes through keychain");
    XCTAssertEqual([attributes[(id)kSecAttrKeyClass] integerValue], [(id)kSecAttrKeyClassPrivate integerValue]);
    XCTAssertEqualObjects(attributes[(id)kSecAttrTokenID], (id)kSecAttrTokenIDAppleKeyStore);
    XCTAssertEqual([attributes[(id)kSecAttrKeyType] integerValue], [(id)kSecAttrKeyTypeECSECPrimeRandom integerValue]);
    XCTAssertEqualObjects(attributes[(id)kSecAttrKeySizeInBits], @256);

    // Create signature with the key.
    NSData *message = [@"message" dataUsingEncoding:NSUTF8StringEncoding];
    SecKeyAlgorithm algorithm = kSecKeyAlgorithmECDSASignatureMessageX962SHA256;
    NSData *signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)privKey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNotNil(signature, @"Failed to sign with token key, error: %@", error);

    // Get public key and verify the signature.
    id pubKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privKey));
    XCTAssertNotNil(pubKey, @"Failed to get pubKey from token privKey");
    XCTAssert(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)message, (CFDataRef)signature, (void *)&error));

    // Perform ECIES encryptoon and decryption.
    algorithm = kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA256AESGCM;
    NSData *ciphertext = CFBridgingRelease(SecKeyCreateEncryptedData((SecKeyRef)pubKey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNotNil(ciphertext, @"Failed to ECIES encrypt data, error:%@", error);

    NSData *plaintext = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privKey, algorithm, (CFDataRef)ciphertext, (void *)&error));
    XCTAssertNotNil(plaintext, @"Failed to decrypt ECIES encrypted data, error:%@", error);
    XCTAssertEqualObjects(plaintext, message);

    // Delete key from keychain.
    query = @{ (id)kSecClass: (id)kSecClassKey, (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore };
    OSStatus status = SecItemDelete((CFDictionaryRef)query);
    XCTAssertEqual(status, errSecSuccess, @"Deletion failed");
}

- (void)testProtectedSecKeyOperations {
    NSData *password = [@"password" dataUsingEncoding:NSUTF8StringEncoding];
    NSError *error;
    id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleWhenUnlocked, kSecAccessControlApplicationPassword | kSecAccessControlPrivateKeyUsage, (void *)&error));
    XCTAssertNotNil(sac);
    LAContext *authContext = [[LAContext alloc] init];
    [authContext setCredential:password type:LACredentialTypeApplicationPassword];

    NSDictionary *attributes = @{
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
        (id)kSecAttrIsPermanent: @YES,
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecUseAuthenticationContext: authContext,
        (id)kSecPrivateKeyAttrs: @{
                (id)kSecAttrAccessControl: sac
        }
    };
    id privKey = CFBridgingRelease(SecKeyCreateRandomKey((CFDictionaryRef)attributes, (void *)&error));
    XCTAssertNotNil(privKey, @"Failed to generate key, error %@", error);

    // Create signature with the key.
    NSData *message = [@"message" dataUsingEncoding:NSUTF8StringEncoding];
    SecKeyAlgorithm algorithm = kSecKeyAlgorithmECDSASignatureMessageX962SHA256;
    NSData *signature = CFBridgingRelease(SecKeyCreateSignature((SecKeyRef)privKey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNotNil(signature, @"Failed to sign with token key, error: %@", error);

    // Get public key and verify the signature.
    id pubKey = CFBridgingRelease(SecKeyCopyPublicKey((SecKeyRef)privKey));
    XCTAssertNotNil(pubKey, @"Failed to get pubKey from token privKey");
    XCTAssert(SecKeyVerifySignature((SecKeyRef)pubKey, algorithm, (CFDataRef)message, (CFDataRef)signature, (void *)&error));

    // Perform ECIES encryptoon and decryption.
    algorithm = kSecKeyAlgorithmECIESEncryptionStandardVariableIVX963SHA256AESGCM;
    NSData *ciphertext = CFBridgingRelease(SecKeyCreateEncryptedData((SecKeyRef)pubKey, algorithm, (CFDataRef)message, (void *)&error));
    XCTAssertNotNil(ciphertext, @"Failed to ECIES encrypt data, error:%@", error);

    NSData *plaintext = CFBridgingRelease(SecKeyCreateDecryptedData((SecKeyRef)privKey, algorithm, (CFDataRef)ciphertext, (void *)&error));
    XCTAssertNotNil(plaintext, @"Failed to decrypt ECIES encrypted data, error:%@", error);
    XCTAssertEqualObjects(plaintext, message);

    // Delete key from keychain.
    NSDictionary *query = @{
        (id)kSecClass: (id)kSecClassKey,
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDAppleKeyStore,
        (id)kSecUseAuthenticationContext: authContext,
    };
    OSStatus status = SecItemDelete((CFDictionaryRef)query);
    XCTAssertEqual(status, errSecSuccess, @"Deletion failed");
}

@end
