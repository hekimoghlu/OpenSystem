/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#import "JVChatWindowController.h"

@class JVChatRoom;
@class NSString;
@class MVChatConnection;
@class JVBuddy;

@interface JVChatRoomMember : NSObject <JVChatListItem> {
	JVChatRoom *_parent;
	NSString *_nickname;
	NSString *_address;
	NSString *_realName;
	JVBuddy *_buddy;
	BOOL _operator;
	BOOL _halfOperator;
	BOOL _serverOperator;
	BOOL _voice;
	
	// Custom ban ivars
	BOOL _nibLoaded;
	IBOutlet NSTextField *banTitle;
	IBOutlet NSTextField *firstTitle;
	IBOutlet NSTextField *secondTitle;
	IBOutlet NSTextField *firstField;
	IBOutlet NSTextField *secondField;
	IBOutlet NSButton *banButton;
	IBOutlet NSButton *cancelButton;
	IBOutlet NSWindow *banWindow;
}
- (id) initWithRoom:(JVChatRoom *) room andNickname:(NSString *) name;

- (NSComparisonResult) compare:(JVChatRoomMember *) member;
- (NSComparisonResult) compareUsingStatus:(JVChatRoomMember *) member;
- (NSComparisonResult) compareUsingBuddyStatus:(JVChatRoomMember *) member;

- (MVChatConnection *) connection;
- (NSString *) nickname;
- (NSString *) realName;
- (NSString *) address;
- (JVBuddy *) buddy;

- (BOOL) voice;
- (BOOL) operator;
- (BOOL) halfOperator;
- (BOOL) serverOperator;
- (BOOL) isLocalUser;

- (IBAction) startChat:(id) sender;
- (IBAction) sendFile:(id) sender;
- (IBAction) addBuddy:(id) sender;

- (IBAction) toggleOperatorStatus:(id) sender;
- (IBAction) toggleVoiceStatus:(id) sender;
- (IBAction) kick:(id) sender;
- (IBAction) ban:(id) sender;
- (IBAction) customKick:(id) sender;
- (IBAction) customBan:(id) sender;
- (IBAction) kickban:(id) sender;
- (IBAction) customKickban:(id) sender;

- (IBAction) closeKickSheet:(id) sender;
- (IBAction) closeBanSheet:(id) sender;
- (IBAction) closeKickbanSheet:(id) sender;
- (IBAction) cancelSheet:(id) sender;
@end

#pragma mark -

@interface JVChatRoomMember (JVChatRoomMemberScripting) <JVChatListItemScripting>
- (NSNumber *) uniqueIdentifier;
@end