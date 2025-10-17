/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

#import <SystemPolicy/SystemPolicy.h>
#import <bootstrap.h>
#import "syspolicy.h"
#import "kext_tools_util.h"

// Basic global state - perform initialization once and rely on that for future calls
static BOOL gInitialized = NO;
static SPKernelExtensionPolicy *gSystemPolicy = nil;

static BOOL
isSystemPolicyLinked() {
    return NSClassFromString(@"SPKernelExtensionPolicy") ? YES : NO;
}

static BOOL
isSystemPolicyServiceAvailable() {
    BOOL serviceIsAvailable = NO;
    mach_port_t syspolicy_port = MACH_PORT_NULL;
    kern_return_t kern_result = 0;
    kern_result = bootstrap_look_up(bootstrap_port,
                                    "com.apple.security.syspolicy.kext",
                                    &syspolicy_port);
    serviceIsAvailable = (kern_result == 0 && syspolicy_port != 0);
    mach_port_deallocate(mach_task_self(), syspolicy_port);
    return serviceIsAvailable;
}

static void
initializeGlobalState() {
    BOOL useSystemPolicy = isSystemPolicyLinked() && isSystemPolicyServiceAvailable();
    if (useSystemPolicy) {
        gSystemPolicy = [[SPKernelExtensionPolicy alloc] init];
    }
    gInitialized = YES;
}

Boolean
SPAllowKextLoad(OSKextRef kext) {
    Boolean allowed = true;

    if (!gInitialized) {
        initializeGlobalState();
    }

    if (gSystemPolicy) {
        NSString *kextPath = (__bridge_transfer NSString*)copyKextPath(kext);
        allowed = [gSystemPolicy canLoadKernelExtension:kextPath error:nil] ? true : false;
    }

    return allowed;
}

Boolean
SPAllowKextLoadCache(OSKextRef kext) {
    Boolean allowed = true;

    if (!gInitialized) {
        initializeGlobalState();
    }

    if (gSystemPolicy) {
        NSString *kextPath = (__bridge_transfer NSString*)copyKextPath(kext);
        allowed = [gSystemPolicy canLoadKernelExtensionInCache:kextPath error:nil] ? true : false;
    }

    return allowed;
}
