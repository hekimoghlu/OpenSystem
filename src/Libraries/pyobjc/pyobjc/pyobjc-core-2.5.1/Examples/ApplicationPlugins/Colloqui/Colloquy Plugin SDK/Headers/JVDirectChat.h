/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

#import "JVChatTranscript.h"
#import <AppKit/NSNibDeclarations.h>
#import <Foundation/NSFileHandle.h>

@class NSView;
@class MVTextView;
@class NSPopUpButton;
@class NSString;
@class NSPanel;
@class MVChatConnection;
@class NSDate;
@class NSMutableData;
@class NSMutableArray;
@class NSMutableDictionary;
@class NSMutableString;
@class NSBundle;
@class NSDictionary;
@class NSToolbar;
@class NSData;
@class NSAttributedString;
@class NSMutableAttributedString;
@class JVBuddy;

@interface JVDirectChat : JVChatTranscript {
	@protected
	IBOutlet MVTextView *send;

	NSString *_target;
	NSStringEncoding _encoding;
	NSMenu *_encodingMenu;
	MVChatConnection *_connection;
	NSMutableArray *_sendHistory;
	NSMutableArray *_waitingAlerts;
	NSMutableDictionary *_waitingAlertNames;
	NSMutableDictionary *_settings;
	NSMenu *_spillEncodingMenu;
	JVBuddy *_buddy;
	NSFileHandle *_logFile;
	NSMutableArray *_messageQueue;

	unsigned int _messageId;
	BOOL _firstMessage;
	BOOL _requiresFullMessage;
	BOOL _isActive;
	unsigned int _newMessageCount;
	unsigned int _newHighlightMessageCount;
	BOOL _cantSendMessages;

	int _historyIndex;	
	float _sendHeight;
	BOOL _scrollerIsAtBottom;
	long _previousLogOffset;
	BOOL _forceSplitViewPosition;
}
- (id) initWithTarget:(NSString *) target forConnection:(MVChatConnection *) connection;

- (void) setTarget:(NSString *) target;
- (NSString *) target;
- (JVBuddy *) buddy;

- (void) unavailable;

- (IBAction) addToFavorites:(id) sender;

- (void) showAlert:(NSPanel *) alert withName:(NSString *) name;

- (void) setPreference:(id) value forKey:(NSString *) key;
- (id) preferenceForKey:(NSString *) key;

- (NSStringEncoding) encoding;
- (IBAction) changeEncoding:(id) sender;	

- (void) addEventMessageToDisplay:(NSString *) message withName:(NSString *) name andAttributes:(NSDictionary *) attributes;
- (void) addEventMessageToDisplay:(NSString *) message withName:(NSString *) name andAttributes:(NSDictionary *) attributes entityEncodeAttributes:(BOOL) encode;
- (void) addMessageToDisplay:(NSData *) message fromUser:(NSString *) user asAction:(BOOL) action;
- (void) processMessage:(NSMutableString *) message asAction:(BOOL) action fromUser:(NSString *) user;
- (void) echoSentMessageToDisplay:(NSAttributedString *) message asAction:(BOOL) action;

- (unsigned int) newMessagesWaiting;
- (unsigned int) newHighlightMessagesWaiting;

- (IBAction) send:(id) sender;
- (void) sendAttributedMessage:(NSMutableAttributedString *) message asAction:(BOOL) action;
- (BOOL) processUserCommand:(NSString *) command withArguments:(NSAttributedString *) arguments;

- (IBAction) clear:(id) sender;
- (IBAction) clearDisplay:(id) sender;
@end

@interface NSObject (MVChatPluginDirectChatSupport)
- (BOOL) processUserCommand:(NSString *) command withArguments:(NSAttributedString *) arguments toChat:(JVDirectChat *) chat;

- (void) processMessage:(NSMutableString *) message asAction:(BOOL) action inChat:(JVDirectChat *) chat;
- (void) processMessage:(NSMutableAttributedString *) message asAction:(BOOL) action toChat:(JVDirectChat *) chat;

- (void) userNamed:(NSString *) nickname isNowKnownAs:(NSString *) newNickname inView:(id <JVChatViewController>) view;
@end
