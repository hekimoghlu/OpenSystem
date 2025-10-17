/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
//  smartcards.m
//  SecurityTool

#import <Foundation/Foundation.h>

#import "smartcards.h"
#import "security_tool.h"

const CFStringRef kTKSmartCardPreferencesDomain = CFSTR("com.apple.security.smartcard");
const CFStringRef kTKDisabledTokensPreferencesKey  = CFSTR("DisabledTokens");

static void listDisabledTokens(void) {
    id value = (__bridge_transfer id)CFPreferencesCopyValue(kTKDisabledTokensPreferencesKey, kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost);
    if (value && ![value isKindOfClass:NSArray.class])
        return;
    NSArray *disabledTokens = (NSArray*)value;
    for (id tokenName in disabledTokens) {
        if ([tokenName isKindOfClass:NSString.class]) {
            printf("\t\"%s\"\n", [tokenName UTF8String]);
        }
    }
}

static void disable(const char *tokenToDisable) {
    id value = (__bridge_transfer id)CFPreferencesCopyValue(kTKDisabledTokensPreferencesKey, kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost);
    if (value && ![value isKindOfClass:NSArray.class])
        return;
    NSMutableArray *disabledTokens = [NSMutableArray arrayWithArray:value];
    NSString *tokenName = [NSString stringWithUTF8String:tokenToDisable];
    if (![disabledTokens containsObject:tokenName]) {
        [disabledTokens addObject:tokenName];
        CFPreferencesSetValue(kTKDisabledTokensPreferencesKey, (__bridge CFTypeRef)disabledTokens, kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost);
        if (!CFPreferencesSynchronize(kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost))
            printf("Permission denied!\n");
    }
    else
        printf("Token is already disabled.\n");
}

static void enable(const char *tokenToEnable) {
    id value = (__bridge_transfer id)CFPreferencesCopyValue(kTKDisabledTokensPreferencesKey, kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost);
    if (value && ![value isKindOfClass:NSArray.class])
        return;
    NSString *tokenName = [NSString stringWithUTF8String:tokenToEnable];
    NSMutableArray *disabledTokens = [NSMutableArray arrayWithArray:value];
    if ([disabledTokens containsObject:tokenName]) {
        [disabledTokens removeObject:tokenName];
        CFPreferencesSetValue(kTKDisabledTokensPreferencesKey, (__bridge CFTypeRef)disabledTokens, kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost);
        if (!CFPreferencesSynchronize(kTKSmartCardPreferencesDomain, kCFPreferencesAnyUser, kCFPreferencesCurrentHost))
            printf("Permission denied!\n");
    }
    else
        printf("Token is already enabled.\n");
}

static int token(int argc, char * const *argv)
{
    int ch;
    while ((ch = getopt(argc, argv, "le:d:")) != -1)
    {
        switch  (ch)
        {
            case 'l':
                listDisabledTokens();
                return 0;
            case 'e':
                enable(optarg);
                return 0;
            case 'd':
                disable(optarg);
                return 0;
        }
    }

    return SHOW_USAGE_MESSAGE;
}

int smartcards(int argc, char * const *argv) {
    int result = 2;
    require_quiet(argc > 2, out);
    @autoreleasepool {
        if (!strcmp("token", argv[1])) {
            result = token(argc - 1, argv + 1);
        }
    }

out:
    return result;
}
