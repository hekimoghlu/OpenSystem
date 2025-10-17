/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
//  migrateenergyprefs.m
//  PowerManagement
//
//  Created by dekom on 1/25/16.
//
//

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <copyfile.h>
#include <IOKit/pwr_mgt/IOPMLibPrivate.h>

#define kSCPrefsFile  "/Library/Preferences/SystemConfiguration/com.apple.PowerManagement.plist"
#define kCFPrefsFile  "/Library/Preferences/com.apple.PowerManagement.plist"

@import SystemMigrationUtils_Private;

@import SystemMigration_Private;

@interface EnergyPrefsMigratorPlugin : SMSystemRulePlugin

-(void)pruneSettings;
@end


@implementation EnergyPrefsMigratorPlugin

-(NSTimeInterval)estimateTime
{
    return 5;
}

-(void)pruneSettings
{
    BOOL dstExists = NO;
    NSURL *file = [NSURL fileURLWithPath:@kCFPrefsFile];

    // CFPrefs settings from source are already copied to target filesystem by the time
    // this plugin is called.
    NSURL *dst = [self.targetFilesystem pathToRemoteFile:file exists:&dstExists makeAvailable:YES];
    if (!dst) {
        SMLog(SMLogItemStatus, @"[PM Migration error] Failed to get URL to destination file\n");
        return;
    }

    NSMutableDictionary *prefs = [[NSMutableDictionary alloc] initWithContentsOfURL: dst];
    if (!prefs) {
        SMLog(SMLogItemStatus, @"[PM Migration error] Failed to create preferences dictionary\n");
        return;
    }
    else {
        SMLog(SMLogItemStatus, @"[PM Migration] Target Preferences: %@\n", prefs);
    }
    bool slider = IOPMFeatureIsAvailable(CFSTR(kIOPMUnifiedSleepSliderPrefKey), CFSTR(kIOPMACPowerKey));
    if (slider) {
        // Remove 'System Sleep' setting when target doesn't support changing that setting from
        // System Preferences application
        NSMutableDictionary *acprefs = prefs[@kIOPMACPowerKey];
        NSMutableDictionary *battprefs = prefs[@kIOPMBatteryPowerKey];
        NSMutableDictionary *upsprefs = prefs[@kIOPMUPSPowerKey];

        if (acprefs) {
            [acprefs removeObjectForKey:@kIOPMSystemSleepKey];
        }
        if (battprefs) {
            [battprefs removeObjectForKey:@kIOPMSystemSleepKey];
        }
        if (upsprefs) {
            [upsprefs removeObjectForKey:@kIOPMSystemSleepKey];
        }
    }
    else {
        SMLog(SMLogItemStatus, @"[PM Migration] Unfied slider is not supported\n");
    }
    
    // Apply the settings on the target system
    IOPMSetPMPreferences((__bridge CFDictionaryRef)prefs);

    SMLog(SMLogItemStatus, @"[PM Migration] Prune settings completed\n");
}

-(void)run
{

    NSURL *dst = NULL;
    BOOL srcExists = NO;
    NSURL *file = [NSURL fileURLWithPath:@kSCPrefsFile];
    NSURL *source = [self.sourceFilesystem pathToRemoteFile:file exists:&srcExists makeAvailable:YES];
    if (!source) {
        SMLog(SMLogItemStatus, @"[PM Migration error] Failed to get the path to SC Prefs file\n");
        goto exit;
    }

    if (!srcExists) {
        SMLog(SMLogItemStatus, @"[PM Migration error] No SC Prefs file found\n");
        goto exit;
    }

    dst = [self.targetFilesystem pathToRemoteFile:file exists:nil makeAvailable:YES];
    if (!dst) {
        SMLog(SMLogItemStatus, @"[PM Migration error] Failed to get the path to destination\n");
        goto exit;
    }

    SMLog(SMLogItemStatus, @"[PM Migration] copying %s to %s", source.fileSystemRepresentation, dst.fileSystemRepresentation);

    if (copyfile(source.fileSystemRepresentation, dst.fileSystemRepresentation, NULL, COPYFILE_DATA | COPYFILE_NOFOLLOW)) {
        SMLog(SMLogItemStatus, @"[PM migration error] Failed to copy file: %s", strerror(errno));
        goto exit;
    }

    SMLog(SMLogItemStatus, @"[PM Migration] Complete");

exit:
    [self pruneSettings];

}

@end
