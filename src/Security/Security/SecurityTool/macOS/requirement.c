/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#include <stdio.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecRequirement.h>
#include <Security/SecRequirementPriv.h>

#include <utilities/SecCFRelease.h>
#include "security_tool.h"
#include "trusted_cert_utils.h"
#include "requirement.h"

int requirement_evaluate(int argc, char * const *argv)
{
    int err = 0;
    CFErrorRef error = NULL;
    CFStringRef reqStr = NULL;
    SecRequirementRef req = NULL;
    CFMutableArrayRef certs = NULL;

    if (argc < 3) {
        return SHOW_USAGE_MESSAGE;
    }

    // Create Requirement
    
    reqStr = CFStringCreateWithCString(NULL, argv[1], kCFStringEncodingUTF8);
    
    OSStatus status = SecRequirementCreateWithStringAndErrors(reqStr,
                                                              kSecCSDefaultFlags, &error, &req);
    
    if (status != errSecSuccess) {
        CFStringRef errorDesc = CFErrorCopyDescription(error);
        CFIndex errorLength = CFStringGetMaximumSizeForEncoding(CFStringGetLength(errorDesc),
                                                                kCFStringEncodingUTF8);
        char *errorStr = malloc(errorLength+1);
        
        CFStringGetCString(errorDesc, errorStr, errorLength+1, kCFStringEncodingUTF8);
        
        fprintf(stderr, "parsing requirement failed (%d): %s\n", status, errorStr);
        
        free(errorStr);
        CFReleaseSafe(errorDesc);
        
        err = 1;
    }

    // Create cert chain
    
    const int num_certs = argc - 2;
    
    certs = CFArrayCreateMutable(NULL, num_certs, &kCFTypeArrayCallBacks);
    
    for (int i = 0; i < num_certs; ++i) {
        SecCertificateRef cert = NULL;
        
        if (readCertFile(argv[2 + i], &cert) != 0) {
            fprintf(stderr, "Error reading certificate at '%s'\n", argv[2 + i]);
            err = 2;
            goto out;
        }
        
        CFArrayAppendValue(certs, cert);
        CFReleaseSafe(cert);
    }
    
    // Evaluate!
    
    if (req != NULL) {
        status = SecRequirementEvaluate(req, certs, NULL, kSecCSDefaultFlags);
        printf("%d\n", status);
        err = status == 0 ? 0 : 3;
    }
    
out:
    CFReleaseSafe(certs);
    CFReleaseSafe(req);
    CFReleaseSafe(reqStr);
    CFReleaseSafe(error);

    return err;
}
