/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
//  secstaticcode_integration.m
//  secsecstaticcodeapitest
//
//  Copyright 2021 Apple Inc. All rights reserved.
//
#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecStaticCode.h>

#import "secstaticcode.h"
#import "codesigning_tests_shared.h"

static void
RevokedBinaryTraversalTest(NSURL *contentRoot)
{
    NSDictionary<NSString *, NSNumber *> *gTestPaths = @{
        // This resource file has a bad signature that will fail validation, but not in a fatal way.
        @"traversal/KV-badsig.app": @(errSecSuccess),
        // These are all hiding revoked binaries in various places for different types of discovery.
        @"traversal/KV-badfile.app": @(CSSMERR_TP_CERT_REVOKED),
        @"traversal/KV-badlink.app": @(CSSMERR_TP_CERT_REVOKED),
        @"traversal/KV-badspot.app": @(CSSMERR_TP_CERT_REVOKED),
    };

    TEST_START("kSecCSEnforceRevocationChecks finds revoked binaries inside bundles");

    for (NSString *path in gTestPaths.allKeys) {
        SecStaticCodeRef codeRef = NULL;
        OSStatus status;

        NSNumber *expected = gTestPaths[path];
        INFO(@"Test case: %@, %@", path, expected);

        NSURL *url = [contentRoot URLByAppendingPathComponent:path];
        status = SecStaticCodeCreateWithPath((__bridge CFURLRef)url, kSecCSDefaultFlags, &codeRef);
        TEST_CASE_EXPR_JUMP(status == errSecSuccess, lb_next);

        status = SecStaticCodeCheckValidity(codeRef, kSecCSEnforceRevocationChecks, NULL);
        INFO(@"validation result: %d", status);
        TEST_CASE(status == expected.integerValue, "validation succeeds with expected result");

lb_next:
        if (codeRef) {
            CFRelease(codeRef);
        }
    }
    return;
}

int
run_integration_tests(const char *root)
{
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:root]];
    NSLog(@"Running integration test with content root: %@", url);

    RevokedBinaryTraversalTest(url);
    return 0;
}
