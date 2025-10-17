/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#import <Security/SecCertificatePriv.h>
#import "TrustFrameworkTestCase.h"
#include "../../../OSX/sec/ipc/securityd_client.h"


@implementation TrustFrameworkTestCase

+ (void)setUp {
    /* XPC to trustd instead of using trustd built-in */
    gTrustd = NULL;
}

- (id _Nullable) CF_RETURNS_RETAINED SecCertificateCreateFromResource:(NSString *)name
                                                         subdirectory:(NSString *)dir
{
    NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:name withExtension:@".cer"
                                                           subdirectory:dir];
    if (!url) {
        url = [[NSBundle bundleForClass:[self class]] URLForResource:name withExtension:@".crt"
                                                        subdirectory:dir];
    }
    NSData *certData = [NSData dataWithContentsOfURL:url];
    if (!certData) {
        return nil;
    }
    SecCertificateRef cert = SecCertificateCreateWithData(kCFAllocatorDefault, (__bridge CFDataRef)certData);
    return (__bridge id)cert;
}

- (id _Nullable) CF_RETURNS_RETAINED SecCertificateCreateFromPEMResource:(NSString *)name
                                                            subdirectory:(NSString *)dir
{
    NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:name withExtension:@".pem"
                                                           subdirectory:dir];
    NSData *certData = [NSData dataWithContentsOfURL:url];
    if (!certData) {
        return nil;
    }

    SecCertificateRef cert = SecCertificateCreateWithPEM(kCFAllocatorDefault, (__bridge CFDataRef)certData);
    return (__bridge id)cert;
}

@end
