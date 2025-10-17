/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#import <AppKit/NSNibDeclarations.h>

@class NSMutableSet;
@class MVChatConnection;
@class NSSet;
@class JVChatWindowController;
@class NSString;
@class JVChatRoom;
@class JVDirectChat;
@class JVChatTranscript;
@class JVChatConsole;
@class NSAttributedString;

@protocol JVChatViewController;

@interface JVChatController : NSObject {
	@private
	NSMutableArray *_chatWindows;
	NSMutableArray *_chatControllers;
	NSMutableDictionary *_ignoreRules;
}
+ (JVChatController *) defaultManager;

- (NSSet *) allChatWindowControllers;
- (JVChatWindowController *) newChatWindowController;
- (void) disposeChatWindowController:(JVChatWindowController *) controller;

- (NSSet *) allChatViewControllers;
- (NSSet *) chatViewControllersWithConnection:(MVChatConnection *) connection;
- (NSSet *) chatViewControllersOfClass:(Class) class;
- (NSSet *) chatViewControllersKindOfClass:(Class) class;
- (JVChatRoom *) chatViewControllerForRoom:(NSString *) room withConnection:(MVChatConnection *) connection ifExists:(BOOL) exists;
- (JVDirectChat *) chatViewControllerForUser:(NSString *) user withConnection:(MVChatConnection *) connection ifExists:(BOOL) exists;
- (JVDirectChat *) chatViewControllerForUser:(NSString *) user withConnection:(MVChatConnection *) connection ifExists:(BOOL) exists userInitiated:(BOOL) requested;
- (JVChatTranscript *) chatViewControllerForTranscript:(NSString *) filename;
- (JVChatConsole *) chatConsoleForConnection:(MVChatConnection *) connection ifExists:(BOOL) exists;

- (void) disposeViewController:(id <JVChatViewController>) controller;
- (void) detachViewController:(id <JVChatViewController>) controller;

- (IBAction) detachView:(id) sender;

- (void) addIgnore:(NSString *)inIgnoreName withKey:(NSString *)ignoreKeyExpression inRooms:(NSArray *) rooms usesRegex:(BOOL) regex isMember:(BOOL) member;
- (BOOL) shouldIgnoreUser:(NSString *) user inRoom:(NSString *) room;
- (BOOL) shouldIgnoreMessage:(NSAttributedString *) message inRoom:(NSString *) room;
- (BOOL) shouldIgnoreMessage:(NSAttributedString *) message fromUser:(NSString *)user inRoom:(NSString *) room;
@end
