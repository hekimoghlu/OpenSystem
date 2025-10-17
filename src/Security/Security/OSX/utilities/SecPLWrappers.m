/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include <TargetConditionals.h>
#include <Foundation/Foundation.h>
#include "SecPLWrappers.h"

#if !TARGET_OS_SIMULATOR && !TARGET_OS_BRIDGE
#include <PowerLog/PowerLog.h>
#include "debugging.h"
#include "sec_action.h"

static typeof(PLShouldLogRegisteredEvent) *soft_PLShouldLogRegisteredEvent = NULL;
static typeof(PLLogRegisteredEvent) *soft_PLLogRegisteredEvent = NULL;
static typeof(PLLogTimeSensitiveRegisteredEvent) *soft_PLLogTimeSensitiveRegisteredEvent = NULL;

static bool gDisabled = false;

static bool
setup(void)
{
    static dispatch_once_t onceToken;
    static sec_action_t action; // for logging whether PowerLog is enabled
    static CFBundleRef bundle = NULL;
    dispatch_once(&onceToken, ^{

        action = sec_action_create("PowerLog enabled", 86400); // log no more than once per day
        sec_action_set_handler(action, ^{
            secnotice("PLsetup", "PowerLog %s", gDisabled ? "disabled" : bundle == NULL ? "fault" : "enabled");
        });

        if (gDisabled) {
            return;
        }

        secnotice("PLsetup", "Setting up PowerLog");
        CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, CFSTR("/System/Library/PrivateFrameworks/PowerLog.framework"), kCFURLPOSIXPathStyle, true);
        if (url == NULL)
            return;

        bundle = CFBundleCreate(kCFAllocatorDefault, url);
        CFRelease(url);
        if (bundle == NULL)
            return;

        soft_PLShouldLogRegisteredEvent = CFBundleGetFunctionPointerForName(bundle, CFSTR("PLShouldLogRegisteredEvent"));
        soft_PLLogRegisteredEvent = CFBundleGetFunctionPointerForName(bundle, CFSTR("PLLogRegisteredEvent"));
        soft_PLLogTimeSensitiveRegisteredEvent = CFBundleGetFunctionPointerForName(bundle, CFSTR("PLLogTimeSensitiveRegisteredEvent"));

        if (soft_PLShouldLogRegisteredEvent == NULL ||
            soft_PLLogRegisteredEvent == NULL ||
            soft_PLLogTimeSensitiveRegisteredEvent == NULL)
        {
            CFRelease(bundle);
            bundle = NULL;
        }
    });

    sec_action_perform(action);

    return bundle != NULL;
}

#endif

void SecPLDisable(void) {
#if !TARGET_OS_SIMULATOR && !TARGET_OS_BRIDGE
    gDisabled = true;
#endif
}

bool SecPLShouldLogRegisteredEvent(NSString *event)
{
#if !TARGET_OS_SIMULATOR && !TARGET_OS_BRIDGE
    if (setup())
        return soft_PLShouldLogRegisteredEvent(PLClientIDSecurity, (__bridge CFStringRef)event);
#endif
    return false;
}

void SecPLLogRegisteredEvent(NSString *eventName, NSDictionary *eventDictionary)
{
#if !TARGET_OS_SIMULATOR && !TARGET_OS_BRIDGE
    if (setup())
        soft_PLLogRegisteredEvent(PLClientIDSecurity,
                                  (__bridge CFStringRef)eventName,
                                  (__bridge CFDictionaryRef)eventDictionary,
                                  NULL);
#endif
}

void SecPLLogTimeSensitiveRegisteredEvent(NSString *eventName, NSDictionary *eventDictionary)
{
#if !TARGET_OS_SIMULATOR && !TARGET_OS_BRIDGE
    if (setup())
        soft_PLLogTimeSensitiveRegisteredEvent(PLClientIDSecurity,
                                               (__bridge CFStringRef)eventName,
                                               (__bridge CFDictionaryRef)eventDictionary,
                                               NULL);
#endif
}

