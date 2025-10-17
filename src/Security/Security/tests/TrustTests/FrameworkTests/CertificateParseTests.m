/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include <AssertMacros.h>
#import <XCTest/XCTest.h>
#include <libDER/oids.h>
#include <Security/SecCertificate.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecCertificateInternal.h>
#include <Security/SecFramework.h>
#include <utilities/SecCFRelease.h>
#include "../TestMacroConversions.h"

#include "TrustFrameworkTestCase.h"

const NSString *kSecTestParseFailureResources = @"si-18-certificate-parse/ParseFailureCerts";
const NSString *kSecTestParseSuccessResources = @"si-18-certificate-parse/ParseSuccessCerts";
const NSString *kSecTestKeyFailureResources = @"si-18-certificate-parse/KeyFailureCerts";
const NSString *kSecTestKeySuccessResources = @"si-18-certificate-parse/KeySuccessCerts";
const NSString *kSecTestTODOFailureResources = @"si-18-certificate-parse/TODOFailureCerts";
const NSString *kSecTestExtensionFailureResources = @"si-18-certificate-parse/ExtensionFailureCerts";
const NSString *kSecTestNameFailureResources = @"si-18-certificate-parse/NameFailureCerts";
const NSString *kSecTrustDuplicateExtensionResources = @"si-18-certificate-parse/DuplicateExtensionCerts";
const NSString *kSecTrustKnownExtensionResources = @"si-18-certificate-parse/KnownExtensionCerts";

@interface CertificateParseTests : TrustFrameworkTestCase

@end

@implementation CertificateParseTests

- (void)testParseFailure {
    /* A bunch of certificates with different parsing errors */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestParseFailureResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test failure certs in bundle.");
    
    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            is(cert, NULL, "Successfully parsed bad cert: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testParseSuccess {
    /* A bunch of certificates with different parsing variations */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestParseSuccessResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test success certs in bundle.");
    
    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Failed to parse good cert: %@", url);
            is(SecCertificateGetUnparseableKnownExtension(cert), kCFNotFound, "Found bad extension in good certs: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testKeyFailure {
    /* Parse failures that require public key extraction */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestKeyFailureResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test key failure certs in bundle.");
    
    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            SecKeyRef pubkey = NULL;
            require_action(cert, blockOut,
                           fail("Failed to parse cert with SPKI error: %@", url));
            pubkey = SecCertificateCopyKey(cert);
            is(pubkey, NULL, "Successfully parsed bad SPKI: %@", url);
            
        blockOut:
            CFReleaseNull(cert);
            CFReleaseNull(pubkey);
        }];
    }
}

- (void)testKeySuccess {
    /* Public keys that should parse */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestKeySuccessResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test key success certs in bundle.");

    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            SecKeyRef pubkey = NULL;
            require_action(cert, blockOut,
                           fail("Failed to parse cert with SPKI error: %@", url));
            pubkey = SecCertificateCopyKey(cert);
            isnt(pubkey, NULL, "Failed to parse cert with good SPKI: %@", url);

        blockOut:
            CFReleaseNull(cert);
            CFReleaseNull(pubkey);
        }];
    }
}

- (void)testTODOFailures {
    /* A bunch of certificates with different parsing errors that currently succeed. */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestTODOFailureResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test TODO failure certs in bundle.");
    
    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Successfully parsed bad TODO cert: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testUnparseableExtensions {
    /* A bunch of certificates with different parsing errors in known (but non-critical) extensions */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestExtensionFailureResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test extension failure certs in bundle.");

    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Failed to parse bad cert with unparseable extension: %@", url);
            isnt(SecCertificateGetUnparseableKnownExtension(cert), kCFNotFound, "Unable to find unparseable extension: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testUnparseableSubjectName {
    /* A bunch of certificates with different parsing errors the subject name */
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTestNameFailureResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test name failure certs in bundle.");

    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Failed to parse bad cert with unparseable name: %@", url);
            is(CFBridgingRelease(SecCertificateCopyCountry(cert)), nil, "Success parsing name for failure cert: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testDuplicateExtensions {
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTrustDuplicateExtensionResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test extension duplicate certs in bundle.");

    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Failed to parse bad cert with duplicate extensions: %@", url);
            isnt(SecCertificateGetDuplicateExtension(cert), kCFNotFound, "Unable to find duplicate extension: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

- (void)testKnownCriticalExtensions {
    NSArray <NSURL *>* certURLs = [[NSBundle bundleForClass:[self class]]URLsForResourcesWithExtension:@".cer" subdirectory:(NSString *)kSecTrustKnownExtensionResources];
    XCTAssertTrue([certURLs count] > 0, "Unable to find parse test extension known certs in bundle.");

    if ([certURLs count] > 0) {
        [certURLs enumerateObjectsUsingBlock:^(NSURL *url, __unused NSUInteger idx, __unused BOOL *stop) {
            NSData *certData = [NSData dataWithContentsOfURL:url];
            SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certData);
            isnt(cert, NULL, "Failed to parse bad cert with duplicate extensions: %@", url);
            XCTAssertFalse(SecCertificateHasUnknownCriticalExtension(cert), "Unable to find duplicate extension: %@", url);
            CFReleaseNull(cert);
        }];
    }
}

@end
