/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include <sys/stat.h>
#include <Security/SecCertificatePriv.h>
#include <utilities/SecFileLocations.h>
#include <utilities/debugging.h>
#include "ipc/securityd_client.h"
#include "trust/trustd/trustd_spi.h"

#import "../TrustEvaluationTestHelpers.h"
#import "TrustDaemonTestCase.h"

@implementation TrustDaemonInitializationTestCase

/* make a new directory for each test case */
static int testNumber = 0;
- (void) setUp {
    NSURL *tmpDirURL = setUpTmpDir();
    tmpDirURL = [tmpDirURL URLByAppendingPathComponent:[NSString stringWithFormat:@"case-%d", testNumber]];

    NSError *error = nil;
    BOOL ok = [[NSFileManager defaultManager] createDirectoryAtURL:tmpDirURL
                                       withIntermediateDirectories:YES
                                                        attributes:NULL
                                                             error:&error];
    if (ok && tmpDirURL) {
        SecSetCustomHomeURL((__bridge CFURLRef)tmpDirURL);
    }
    testNumber++;
    gTrustd = &trustd_spi; // Signal that we're running as (uninitialized) trustd

    /* Because each test case gets a new "home" directory but we only create the data vault hierarchy once per
     * launch, we need to initialize those directories for each test case. */
    WithPathInProtectedDirectory(CFSTR("trustd"), ^(const char *path) {
        mode_t permissions = 0755; // Non-system trustd's create directory with expansive permissions
        int ret = mkpath_np(path, permissions);
        chmod(path, permissions);
        if (!(ret == 0 || ret ==  EEXIST)) {
            secerror("could not create path: %s (%s)", path, strerror(ret));
        }
    });
}
@end

@implementation TrustDaemonTestCase

/* Build in trustd functionality to the tests */
+ (void) setUp {
    NSURL *tmpDirURL = setUpTmpDir();
    trustd_init((__bridge CFURLRef) tmpDirURL);

    // "Disable" evaluation analytics (by making the sampling rate as low as possible)
    NSUserDefaults *defaults = [[NSUserDefaults alloc] initWithSuiteName:@"com.apple.security"];
    [defaults setInteger:INT32_MAX forKey:@"TrustEvaluationEventAnalyticsRate"];
    [defaults setInteger:INT32_MAX forKey:@"PinningEventAnalyticsRate"];
    [defaults setInteger:INT32_MAX forKey:@"SystemRootUsageEventAnalyticsRate"];
    [defaults setInteger:INT32_MAX forKey:@"TrustFailureEventAnalyticsRate"];
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
