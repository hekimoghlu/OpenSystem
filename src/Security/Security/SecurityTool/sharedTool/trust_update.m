/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#import <Foundation/Foundation.h>

#import <utilities/SecCFWrappers.h>
#import <Security/SecTrustPriv.h>

#include "SecurityCommands.h"

static int check_OTA_Supplementals_asset(void) {
    CFErrorRef error = NULL;
    uint64_t version = SecTrustOTAPKIGetUpdatedAsset(&error);
    if (error) {
        CFStringRef errorDescription = CFErrorCopyDescription(error);
        if (errorDescription) {
            char *errMsg = CFStringToCString(errorDescription);
            fprintf(stdout, "Update failed: %s\n", errMsg);
            if (errMsg) { free(errMsg); }
            CFRelease(errorDescription);
        } else {
            fprintf(stdout, "Update failed: no description\n");
        }
        CFRelease(error);
    } else {
        fprintf(stdout, "Updated succeeded\n");
    }
    if (version != 0) {
        fprintf(stdout, "Asset Content Version: %llu\n", version);
    } else {
        return 1;
    }
    return 0;
}

static int check_OTA_sec_experiment_asset(void) {
    CFErrorRef error = NULL;
    uint64_t version = SecTrustOTASecExperimentGetUpdatedAsset(&error);
    if (error) {
        CFStringRef errorDescription = CFErrorCopyDescription(error);
        if (errorDescription) {
            char *errMsg = CFStringToCString(errorDescription);
            fprintf(stdout, "Update failed: %s\n", errMsg);
            if (errMsg) { free(errMsg); }
            CFRelease(errorDescription);
        } else {
            fprintf(stdout, "Update failed: no description\n");
        }
        CFRelease(error);
    } else {
        fprintf(stdout, "Updated succeeded\n");
    }
    if (version != 0) {
        fprintf(stdout, "Asset Content Version: %llu\n", version);
    } else {
        return 1;
    }
    return 0;
}

static int check_valid_update(void) {
    CFErrorRef error = NULL;
    bool result = SecTrustTriggerValidUpdate(&error);
    if (!result) {
        CFStringRef errorDescription = error ? CFErrorCopyDescription(error) : NULL;
        if (errorDescription) {
            char *errMsg = CFStringToCString(errorDescription);
            fprintf(stdout, "Update failed: %s\n", errMsg ? errMsg : "no error message");
            free(errMsg);
            CFRelease(errorDescription);
        } else {
            fprintf(stdout, "Update failed: no description\n");
        }
        CFReleaseNull(error);
    } else {
        fprintf(stdout, "Updated triggered\n");
    }
    return 0;
}

int check_trust_update(int argc, char * const *argv) {
    int arg;

    if (argc == 1) {
        return SHOW_USAGE_MESSAGE;
    }

    while ((arg = getopt(argc, argv, "ser")) != -1) {
        switch(arg) {
            case 's':
                return check_OTA_Supplementals_asset();
            case 'e':
                return check_OTA_sec_experiment_asset();
            case 'r':
                return check_valid_update();
            case '?':
            default:
                return SHOW_USAGE_MESSAGE;
        }
    }

    return 0;
}
