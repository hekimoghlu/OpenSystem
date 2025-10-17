/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

#import <Foundation/NSObject.h>
#import <Foundation/NSDate.h>

extern NSString *JVBuddyCameOnlineNotification;
extern NSString *JVBuddyWentOfflineNotification;

extern NSString *JVBuddyNicknameCameOnlineNotification;
extern NSString *JVBuddyNicknameWentOfflineNotification;
extern NSString *JVBuddyNicknameStatusChangedNotification;

extern NSString *JVBuddyActiveNicknameChangedNotification;

@class ABPerson;
@class NSMutableSet;
@class NSMutableDictionary;
@class NSURL;
@class NSString;
@class NSImage;
@class NSSet;

typedef enum {
	JVBuddyOfflineStatus = 'oflN',
	JVBuddyAvailableStatus = 'avaL',
	JVBuddyIdleStatus = 'idlE',
	JVBuddyAwayStatus = 'awaY'
} JVBuddyStatus;

typedef enum {
	JVBuddyActiveNickname = 0x0,
	JVBuddyGivenNickname = 0x1,
	JVBuddyFullName = 0x2
} JVBuddyName;

@interface JVBuddy : NSObject {
	ABPerson *_person;
	NSMutableSet *_nicknames;
	NSMutableSet *_onlineNicknames;
	NSMutableDictionary *_nicknameStatus;
	NSURL *_activeNickname;
}
+ (JVBuddyName) preferredName;
+ (void) setPreferredName:(JVBuddyName) preferred;

+ (id) buddyWithPerson:(ABPerson *) person;
+ (id) buddyWithUniqueIdentifier:(NSString *) identifier;

- (id) initWithPerson:(ABPerson *) person;

- (void) registerWithApplicableConnections;
- (void) unregisterWithApplicableConnections;

- (NSURL *) activeNickname;
- (void) setActiveNickname:(NSURL *) nickname;

- (JVBuddyStatus) status;
- (BOOL) isOnline;
- (NSTimeInterval) idleTime;
- (NSString *) awayMessage;

- (NSSet *) nicknames;
- (NSSet *) onlineNicknames;

- (void) addNickname:(NSURL *) nickname;
- (void) removeNickname:(NSURL *) nickname;
- (void) replaceNickname:(NSURL *) old withNickname:(NSURL *) new;

- (NSImage *) picture;
- (void) setPicture:(NSImage *) picture;

- (NSString *) preferredName;
- (JVBuddyName) preferredNameWillReturn;
- (unsigned int) availableNames;

- (NSString *) compositeName;
- (NSString *) firstName;
- (NSString *) lastName;
- (NSString *) primaryEmail;
- (NSString *) givenNickname;

- (void) setFirstName:(NSString *) name;
- (void) setLastName:(NSString *) name;
- (void) setPrimaryEmail:(NSString *) email;
- (void) setGivenNickname:(NSString *) name;

- (NSString *) uniqueIdentifier;
- (ABPerson *) person;
- (void) editInAddressBook;
- (void) viewInAddressBook;

- (NSComparisonResult) availabilityCompare:(JVBuddy *) buddy;
- (NSComparisonResult) firstNameCompare:(JVBuddy *) buddy;
- (NSComparisonResult) lastNameCompare:(JVBuddy *) buddy;
- (NSComparisonResult) serverCompare:(JVBuddy *) buddy;
- (NSComparisonResult) nicknameCompare:(JVBuddy *) buddy;
@end
