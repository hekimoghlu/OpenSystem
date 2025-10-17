/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#import "config.h"
#import "_WKSystemPreferencesInternal.h"

#import <wtf/Assertions.h>
#import <wtf/RetainPtr.h>

#if HAVE(LOCKDOWN_MODE_FRAMEWORK)
#import <pal/cocoa/LockdownModeCocoa.h>
#endif

constexpr auto CaptivePortalConfigurationIgnoreFileName = @"com.apple.WebKit.cpmconfig_ignore";

@implementation _WKSystemPreferences

+ (BOOL)isCaptivePortalModeEnabled
{
    auto preferenceValue = adoptCF(CFPreferencesCopyValue(WKLockdownModeEnabledKeyCFString, kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost));
    if (preferenceValue.get() == kCFBooleanTrue)
        return true;

#if HAVE(LOCKDOWN_MODE_FRAMEWORK)
    return PAL::isLockdownModeEnabled();
#else
    preferenceValue = adoptCF(CFPreferencesCopyValue(LDMEnabledKey, kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost));
    return preferenceValue.get() == kCFBooleanTrue;
#endif
}

+ (void)setCaptivePortalModeEnabled:(BOOL)enabled
{
    CFPreferencesSetValue(WKLockdownModeEnabledKeyCFString, enabled ? kCFBooleanTrue : kCFBooleanFalse, kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost);
    CFPreferencesSynchronize(kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost);
    CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), (__bridge CFStringRef)WKLockdownModeContainerConfigurationChangedNotification, nullptr, nullptr, true);
}

+ (BOOL)isCaptivePortalModeIgnored:(NSString *)containerPath
{
#if PLATFORM(IOS_FAMILY)
    NSString *cpmconfigIgnoreFilePath = [NSString pathWithComponents:@[containerPath, @"System/Preferences/", CaptivePortalConfigurationIgnoreFileName]];
    return [[NSFileManager defaultManager] fileExistsAtPath:cpmconfigIgnoreFilePath];
#endif
    return false;
}

+ (void)setCaptivePortalModeIgnored:(NSString *)containerPath ignore:(BOOL)ignore
{
#if PLATFORM(IOS_FAMILY)
    NSString *cpmconfigDirectoryPath = [NSString pathWithComponents:@[containerPath, @"System/Preferences/"]];
    NSString *cpmconfigIgnoreFilePath = [NSString pathWithComponents:@[cpmconfigDirectoryPath, CaptivePortalConfigurationIgnoreFileName]];
    if ([[NSFileManager defaultManager] fileExistsAtPath:cpmconfigIgnoreFilePath] == ignore)
        return;

    BOOL cpmconfigDirectoryisDir;
    BOOL cpmconfigDirectoryPathExists = [[NSFileManager defaultManager] fileExistsAtPath:cpmconfigDirectoryPath isDirectory:&cpmconfigDirectoryisDir];

    if (!cpmconfigDirectoryisDir)
        [[NSFileManager defaultManager] removeItemAtPath:cpmconfigDirectoryPath error:NULL];

    if (!cpmconfigDirectoryPathExists || !cpmconfigDirectoryisDir)
        [[NSFileManager defaultManager] createDirectoryAtPath:cpmconfigDirectoryPath withIntermediateDirectories:YES attributes:NULL error:NULL];

    if (ignore)
        [[NSFileManager defaultManager] createFileAtPath:cpmconfigIgnoreFilePath contents:NULL attributes:NULL];
    else
        [[NSFileManager defaultManager] removeItemAtPath:cpmconfigIgnoreFilePath error:NULL];

    CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), (__bridge CFStringRef)WKLockdownModeContainerConfigurationChangedNotification, nullptr, nullptr, true);
#endif
}

@end
