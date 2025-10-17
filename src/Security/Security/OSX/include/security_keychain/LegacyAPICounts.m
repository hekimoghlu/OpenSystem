/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#import <os/feature_private.h>
#import <sys/codesign.h>

#import "LegacyAPICounts.h"
#import "utilities/SecCoreAnalytics.h"
#import "SecEntitlements.h"
#import "debugging.h"
#import "SecInternalReleasePriv.h"

#pragma mark - File-Private

// NOTE: LegacyAPICounts is default enabled!!

static NSString* applicationIdentifierForSelf(void) {
    NSString* identifier = nil;
    SecTaskRef task = SecTaskCreateFromSelf(kCFAllocatorDefault);

    if (task) {
        CFStringRef val = (CFStringRef)SecTaskCopyValueForEntitlement(task, kSecEntitlementApplicationIdentifier, NULL);
        if (val && CFGetTypeID(val) != CFStringGetTypeID()) {
            CFRelease(val);
        } else {
            identifier = CFBridgingRelease(val);
        }

        if (!identifier) {
            CFBundleRef mainbundle = CFBundleGetMainBundle();
            if (mainbundle != NULL) {
                CFStringRef tmp = CFBundleGetIdentifier(mainbundle);
                if (tmp != NULL) {
                    identifier = (__bridge NSString*)tmp;
                }
            }
        }

        if (!identifier) {
            identifier = CFBridgingRelease(SecTaskCopySigningIdentifier(task, NULL));
        }

        if (!identifier) {
            identifier = [NSString stringWithCString:getprogname() encoding:NSUTF8StringEncoding];
        }

        CFRelease(task);
    }

    return identifier;
}

static NSString* identifier;

static void setup(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        identifier = applicationIdentifierForSelf() ?: @"unknown";
    });
}

#pragma mark - SPI

void setCountLegacyAPIEnabledForThreadCleanup(bool* value) {
    if(value != NULL) {
        setCountLegacyAPIEnabledForThread(*value);
    }
}

void setCountLegacyAPIEnabledForThread(bool value) {
    [[NSThread currentThread] threadDictionary][@"countLegacyAPIEnabled"] = value ? @YES : @NO;
}

bool countLegacyAPIEnabledForThread(void) {
    NSNumber* value = [[NSThread currentThread] threadDictionary][@"countLegacyAPIEnabled"];

    // No value means not set at all, so not disabled by SecItem*
    if (!value || (value && [value isKindOfClass:[NSNumber class]] && [value boolValue])) {
        return true;
    }
    return false;
}

void countLegacyAPI(dispatch_once_t* token, const char* api) {
    setup();

    if (api == nil) {
        secerror("LegacyAPICounts: Attempt to count API without name");
        return;
    }

    if (!countLegacyAPIEnabledForThread()) {
        return;
    }

    dispatch_once(token, ^{
        NSString* apiStringObject = [NSString stringWithCString:api encoding:NSUTF8StringEncoding];
        if (!apiStringObject) {
            secerror("LegacyAPICounts: Surprisingly, char* for api name \"%s\" did not turn into NSString", api);
            return;
        }

        [SecCoreAnalytics sendEventLazy:@"com.apple.security.LegacyAPICounts" builder:^NSDictionary<NSString *,NSObject *> * _Nonnull{
            return @{
                @"app" : identifier,
                @"api" : apiStringObject,
            };
        }];
    });
}

void countLegacyMDSPlugin(const char* path, const char* guid) {
    setup();
    
    NSString* pathString = [NSString stringWithCString:path encoding:NSUTF8StringEncoding];
    if (!pathString) {
        secerror("LegacyAPICounts: Unable to make NSString from path %s", path);
        return;
    }

    NSString* guidString = [NSString stringWithCString:guid encoding:NSUTF8StringEncoding];
    if (!guidString) {
        secerror("LegacyAPICounts: Unable to make NSString from guid %s", guid);
        return;
    }

    if(path && *path == '*') {
        // These are apparently 'built-in psuedopaths'. Don't log.
        secinfo("mds", "Ignoring the built-in MDS plugin: %@ %@", pathString, guidString);

    } else {
        secnotice("mds", "Recording an MDS plugin: %@ %@", pathString, guidString);

        [SecCoreAnalytics sendEventLazy:@"com.apple.security.LegacyMDSPluginCounts" builder:^NSDictionary<NSString *,NSObject *> * _Nonnull{
            return @{
                @"app" : identifier,
                @"mdsPath" : pathString,
                @"mdsGuid" : guidString,
            };
        }];
    }
}
