/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#import "KNPersistentState.h"
#import <utilities/debugging.h>

@implementation KNPersistentState

-(NSURL*)urlForStorage
{
	return [NSURL URLWithString:@"Preferences/com.apple.security.KCN.plist" relativeToURL:[[NSFileManager defaultManager] URLForDirectory:NSLibraryDirectory inDomain:NSUserDomainMask appropriateForURL:nil create:YES error:nil]];
}

+(instancetype)loadFromStorage
{
	KNPersistentState *state = [KNPersistentState new];
    if (!state) {
        return state;
    }
    
    id plist = @{@"lastWritten": [NSDate distantPast]};

    NSError *error = nil;
    NSData *stateData = [NSData dataWithContentsOfURL:[state urlForStorage] options:0 error:&error];
    if (!stateData) {
        secdebug("kcn", "Can't read state data (p=%@, err=%@)", [state urlForStorage], error);
    } else {
        NSPropertyListFormat format;
        plist = [NSPropertyListSerialization propertyListWithData:stateData options: NSPropertyListMutableContainersAndLeaves format:&format error:&error];
        
        if (plist == nil) {
            secdebug("kcn", "Can't deserialize %@, e=%@", stateData, error);
        }
    }
    
    state.lastCircleStatus 						= plist[@"lastCircleStatus"] ? [plist[@"lastCircleStatus"] intValue] : kSOSCCCircleAbsent;
    state.lastWritten							= plist[@"lastWritten"];
	state.pendingApplicationReminder			= plist[@"pendingApplicationReminder"] ?: [NSDate distantFuture];
	state.applicationDate						= plist[@"applicationDate"]            ?: [NSDate distantPast];
	state.debugLeftReason						= plist[@"debugLeftReason"];
	state.pendingApplicationReminderInterval	= plist[@"pendingApplicationReminderInterval"];
	state.absentCircleWithNoReason				= plist[@"absentCircleWithNoReason"] ? [plist[@"absentCircleWithNoReason"] intValue] : NO;
    state.applicantNotificationTimestamp        = plist[@"applicantNotificationTimestamp"] ?: [NSDate distantPast];

    if (!state.pendingApplicationReminderInterval || [state.pendingApplicationReminderInterval doubleValue] <= 0) {
        state.pendingApplicationReminderInterval = [NSNumber numberWithUnsignedInt: 24*60*60];
    }
    
    return state;
}

-(void)writeToStorage
{
    NSMutableDictionary *plist = [@{@"lastCircleStatus"					 : [NSNumber numberWithInt:self.lastCircleStatus],
									@"lastWritten"						 : [NSDate date],
									@"applicationDate"					 : self.applicationDate,
									@"pendingApplicationReminder"		 : self.pendingApplicationReminder,
									@"pendingApplicationReminderInterval": self.pendingApplicationReminderInterval,
									@"absentCircleWithNoReason" 		 : [NSNumber numberWithBool:self.absentCircleWithNoReason],
                                    @"applicantNotificationTimestamp"    : self.applicantNotificationTimestamp ?: [NSDate distantPast]
								   } mutableCopy];
	if (self.debugLeftReason)
		plist[@"debugLeftReason"] = self.debugLeftReason;
    secdebug("kcn", "writeToStorage plist=%@", plist);
	
    NSError *error = nil;
    NSData *stateData = [NSPropertyListSerialization dataWithPropertyList:plist format:NSPropertyListXMLFormat_v1_0 options:kCFPropertyListImmutable error:&error];
    if (!stateData) {
        secdebug("kcn", "Can't serialize %@: %@", plist, error);
        return;
    }
    if (![stateData writeToURL:[self urlForStorage] options:NSDataWritingAtomic error:&error]) {
        secdebug("kcn", "Can't write to %@, error=%@", [self urlForStorage], error);
    }
}


@end
