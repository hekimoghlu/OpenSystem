/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
//  main.m
//  test_pam_localauthentication
//
//  Created by Jiri Margaritov on 25/11/15.
//
//

#import <Foundation/Foundation.h>

#include <security/pam_appl.h>
#include <security/openpam.h>
#import <LocalAuthentication/LAContext+Private.h>

int main(int argc, const char * argv[]) {
    int pam_res = PAM_SYSTEM_ERR;

    @autoreleasepool {
        pam_handle_t *pamh = NULL;
        struct pam_conv pamc = { openpam_nullconv, NULL };
        LAContext *context = nil;
        CFDataRef econtext = NULL;
        const char *username = NULL;

        if (argc > 1) {
            username = argv[1];
        }

        if (!username)
            goto cleanup;

        if (PAM_SUCCESS != (pam_res = pam_start("localauthentication", username, &pamc, &pamh)))
            goto cleanup;

        context = [LAContext new];
        econtext = CFDataCreate(kCFAllocatorDefault, context.externalizedContext.bytes, context.externalizedContext.length);

        if (!econtext)
            goto cleanup;

        if (PAM_SUCCESS != (pam_res = pam_set_data(pamh, "token_la", (void *)&econtext, NULL)))
            goto cleanup;

        if (PAM_SUCCESS != (pam_res = pam_authenticate(pamh, 0)))
            goto cleanup;
        
        if (PAM_SUCCESS != (pam_res = pam_acct_mgmt(pamh, 0)))
            goto cleanup;

cleanup:
        if (pamh) {
            pam_end(pamh, pam_res);
        }
        
        if (econtext) {
            CFRelease(econtext);
        }
    }

    return pam_res;
}
