/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#ifndef LocalKeychainAnalytics_h
#define LocalKeychainAnalytics_h

#include <CoreFoundation/CoreFoundation.h>

typedef enum {
    LKAKeychainUpgradeOutcomeSuccess,
    LKAKeychainUpgradeOutcomeUnknownFailure,
    LKAKeychainUpgradeOutcomeLocked,
    LKAKeychainUpgradeOutcomeInternal,
    LKAKeychainUpgradeOutcomeNewDb,
    LKAKeychainUpgradeOutcomeObsoleteDb,
    LKAKeychainUpgradeOutcomeNoSchema,
    LKAKeychainUpgradeOutcomeIndices,
    LKAKeychainUpgradeOutcomePhase1AlterTables,
    LKAKeychainUpgradeOutcomePhase1DropIndices,
    LKAKeychainUpgradeOutcomePhase1CreateSchema,
    LKAKeychainUpgradeOutcomePhase1Items,
    LKAKeychainUpgradeOutcomePhase1NonItems,
    LKAKeychainUpgradeOutcomePhase1DropOld,
    LKAKeychainUpgradeOutcomePhase2,
} LKAKeychainUpgradeOutcome;

typedef enum {
    LKAKeychainBackupTypeNeither,
    LKAKeychainBackupTypeBag,
    LKAKeychainBackupTypeCode,
    LKAKeychainBackupTypeBagAndCode,
    LKAKeychainBackupTypeEMCS,
} LKAKeychainBackupType;

void LKAReportKeychainUpgradeOutcome(int fromversion, int toversion, LKAKeychainUpgradeOutcome outcome);
void LKAReportKeychainUpgradeOutcomeWithError(int fromversion, int toversion, LKAKeychainUpgradeOutcome outcome, CFErrorRef error);

void LKABackupReportStart(bool hasKeybag, bool hasPasscode, bool isEMCS);
void LKABackupReportEnd(bool hasBackup, CFErrorRef error);

// For tests only
void LKAForceClose(void);

#if __OBJC2__

#import <Foundation/Foundation.h>
#import <Security/SFAnalytics.h>

typedef NSString* LKAnalyticsFailableEvent NS_STRING_ENUM;
typedef NSString* LKAnalyticsMetric NS_STRING_ENUM;

extern LKAnalyticsFailableEvent const LKAEventUpgrade;
extern LKAnalyticsFailableEvent const LKAEventStash;
extern LKAnalyticsFailableEvent const LKAEventStashLoad;

@interface LocalKeychainAnalytics : SFAnalytics

- (void)reportKeychainUpgradeFrom:(int)oldVersion to:(int)newVersion outcome:(LKAKeychainUpgradeOutcome)result error:(NSError*)error;

@end

#endif  // OBJC2
#endif  // LocalKeychainAnalytics_h
