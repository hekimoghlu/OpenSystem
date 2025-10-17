/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#import <AppKit/NSWindowController.h>
#import <AppKit/NSNibDeclarations.h>
#import "JVInspectorController.h"

@class NSTableView;
@class NSWindow;
@class NSPanel;
@class NSTextField;
@class NSPopUpButton;
@class NSButton;
@class NSTabView;
@class NSComboBox;
@class NSString;
@class NSMutableArray;
@class MVChatConnection;
@class NSURL;

@interface MVConnectionsController : NSWindowController <JVInspectionDelegator> {
@private
	IBOutlet NSTableView *connections;
	IBOutlet NSPanel *openConnection;
	IBOutlet NSPanel *messageUser;
	IBOutlet NSPanel *nicknameAuth;

	/* Nick Auth */
	IBOutlet NSTextField *authNickname;
	IBOutlet NSTextField *authAddress;
	IBOutlet NSTextField *authPassword;
	IBOutlet NSButton *authKeychain;

	/* New Connection */
	IBOutlet NSTextField *newNickname;
	IBOutlet NSTextField *newAddress;
	IBOutlet NSTextField *newPort;
	IBOutlet NSButton *newRemember;
	IBOutlet NSButton *showDetails;
	IBOutlet NSTabView *detailsTabView;
	IBOutlet NSTextField *newServerPassword;
	IBOutlet NSTextField *newUsername;
	IBOutlet NSTextField *newRealName;
	IBOutlet NSPopUpButton *newProxy;
	IBOutlet NSTableView *newJoinRooms;
	IBOutlet NSButton *newRemoveRoom;
	IBOutlet NSButton *sslConnection;

	/* Message User */
	IBOutlet NSTextField *userToMessage;

	NSMutableArray *_bookmarks;
	NSMutableArray *_joinRooms;
	MVChatConnection *_passConnection;
}
+ (MVConnectionsController *) defaultManager;

+ (NSMenu *) favoritesMenu;
+ (void) refreshFavoritesMenu;

- (IBAction) showConnectionManager:(id) sender;
- (IBAction) hideConnectionManager:(id) sender;

- (IBAction) newConnection:(id) sender;
- (IBAction) toggleNewConnectionDetails:(id) sender;
- (IBAction) addRoom:(id) sender;
- (IBAction) removeRoom:(id) sender;
- (IBAction) openNetworkPreferences:(id) sender;
- (IBAction) conenctNewConnection:(id) sender;

- (IBAction) messageUser:(id) sender;

- (IBAction) sendPassword:(id) sender;

- (NSArray *) connections;
- (NSArray *) connectedConnections;
- (MVChatConnection *) connectionForServerAddress:(NSString *) address;
- (NSArray *) connectionsForServerAddress:(NSString *) address;

- (void) setAutoConnect:(BOOL) autoConnect forConnection:(MVChatConnection *) connection;
- (BOOL) autoConnectForConnection:(MVChatConnection *) connection;

- (void) setJoinRooms:(NSArray *) rooms forConnection:(MVChatConnection *) connection;
- (NSArray *) joinRoomsForConnection:(MVChatConnection *) connection;

- (void) setConnectCommands:(NSString *) commands forConnection:(MVChatConnection *) connection;
- (NSString *) connectCommandsForConnection:(MVChatConnection *) connection;

- (void) addConnection:(MVChatConnection *) connection;
- (void) addConnection:(MVChatConnection *) connection keepBookmark:(BOOL) keep;
- (void) insertConnection:(MVChatConnection *) connection atIndex:(unsigned) index;
- (void) removeConnectionAtIndex:(unsigned) index;
- (void) replaceConnectionAtIndex:(unsigned) index withConnection:(MVChatConnection *) connection;

- (void) handleURL:(NSURL *) url andConnectIfPossible:(BOOL) connect;
@end