/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#import "JVChatWindowController.h"

@class NSView;
@class NSTextView;
@class MVTextView;
@class MVChatConnection;

@interface JVChatConsole : NSObject <JVChatViewController> {
	@protected
	IBOutlet NSView *contents;
	IBOutlet NSTextView *display;
	IBOutlet MVTextView *send;
	BOOL _nibLoaded;
	BOOL _verbose;
	BOOL _ignorePRIVMSG;
	BOOL _paused;
	float _sendHeight;
	BOOL _scrollerIsAtBottom;
	BOOL _forceSplitViewPosition;
	int _historyIndex;
	NSMutableArray *_sendHistory;
	JVChatWindowController *_windowController;
	MVChatConnection *_connection;
}
- (id) initWithConnection:(MVChatConnection *) connection;

- (void) pause;
- (void) resume;
- (BOOL) isPaused;
	
- (void) addMessageToDisplay:(NSString *) message asOutboundMessage:(BOOL) outbound;
- (IBAction) send:(id) sender;
@end

@interface JVChatConsole (JVChatConsoleScripting) <JVChatListItemScripting>
- (NSNumber *) uniqueIdentifier;
@end

@interface NSObject (MVChatPluginCommandSupport)
- (BOOL) processUserCommand:(NSString *) command withArguments:(NSAttributedString *) arguments toConnection:(MVChatConnection *) connection;
@end
