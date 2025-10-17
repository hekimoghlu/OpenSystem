/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#include "shared_regressions.h"

#import <AssertMacros.h>
#import <Foundation/Foundation.h>

#include <Security/SecCmsMessage.h>
#include <Security/SecCmsDecoder.h>

const NSString *kSecTestCMSParseFailureResources = @"si-27-cms-parse/ParseFailureCMS";
const NSString *kSecTestCMSParseSuccessResources = @"si-27-cms-parse/ParseSuccessCMS";


static void test_cms_parse_success(void) {
    NSArray<NSURL*>*cmsURLs = [[NSBundle mainBundle] URLsForResourcesWithExtension:@".der" subdirectory:(NSString *)kSecTestCMSParseSuccessResources];
    if ([cmsURLs count] > 0) {
        [cmsURLs enumerateObjectsUsingBlock:^(NSURL * _Nonnull url, NSUInteger __unused idx, BOOL * __unused _Nonnull stop) {
            SecCmsMessageRef cmsg = NULL;
            NSData *cmsData = [NSData dataWithContentsOfURL:url];
            SecAsn1Item encoded_message = { [cmsData length], (uint8_t*)[cmsData bytes] };
            ok_status(SecCmsMessageDecode(&encoded_message, NULL, NULL, NULL, NULL, NULL, NULL, &cmsg), "Failed to parse CMS: %@", url);
            if (cmsg) SecCmsMessageDestroy(cmsg);
        }];
    }
}

static void test_cms_parse_failure(void) {
    NSArray<NSURL*>*cmsURLs = [[NSBundle mainBundle] URLsForResourcesWithExtension:@".der" subdirectory:(NSString *)kSecTestCMSParseFailureResources];
    if ([cmsURLs count] > 0) {
        [cmsURLs enumerateObjectsUsingBlock:^(NSURL * _Nonnull url, NSUInteger __unused idx, BOOL * __unused _Nonnull stop) {
            SecCmsMessageRef cmsg = NULL;
            NSData *cmsData = [NSData dataWithContentsOfURL:url];
            SecAsn1Item encoded_message = { [cmsData length], (uint8_t*)[cmsData bytes] };
            isnt(errSecSuccess, SecCmsMessageDecode(&encoded_message, NULL, NULL, NULL, NULL, NULL, NULL, &cmsg),
                 "Successfully parsed bad CMS: %@", url);
            if (cmsg) SecCmsMessageDestroy(cmsg);
        }];
    }
}

int si_27_cms_parse(int argc, char *const *argv)
{
    int num_tests = 1;
    NSArray<NSURL*>*cmsURLs = [[NSBundle mainBundle] URLsForResourcesWithExtension:@".der" subdirectory:(NSString *)kSecTestCMSParseFailureResources];
    num_tests += [cmsURLs count];
    cmsURLs = [[NSBundle mainBundle] URLsForResourcesWithExtension:@".der" subdirectory:(NSString *)kSecTestCMSParseSuccessResources];
    num_tests += [cmsURLs count];

    plan_tests(num_tests);
    isnt(num_tests, 1, "no tests run!");

    test_cms_parse_success();
    test_cms_parse_failure();

    return 0;
}
