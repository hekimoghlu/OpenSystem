/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#import <Foundation/Foundation.h>
#import "JVAutoEmotionDisplay.h"
#import "JVChatRoom.h"
#import "JVDirectChat.h"

@implementation JVAutoEmotionDisplay
- (id) initWithManager:(MVChatPluginManager *) manager {
	self = [super init];
	_manager = manager; // Don't retain, to prevent a circular retain.
	_emotions = [[NSMutableDictionary dictionary] retain];
	return self;
}

- (void) dealloc {
	[_emotions release];
	_emotions = nil;
	_manager = nil;
	[super dealloc];	
}

- (BOOL) processUserCommand:(NSString *) command withArguments:(NSAttributedString *) arguments toRoom:(JVChatRoom *) room {
	if( [command isEqualToString:@"aed"] ) {
		if( arguments ) [_emotions setObject:[[arguments copy] autorelease] forKey:[room target]];
		else [_emotions removeObjectForKey:[room target]];
		return YES;
	}
	return NO;
}

- (BOOL) processUserCommand:(NSString *) command withArguments:(NSAttributedString *) arguments toChat:(JVDirectChat *) chat {
	if( [command isEqualToString:@"aed"] ) {
		if( arguments ) [_emotions setObject:[[arguments copy] autorelease] forKey:[chat target]];
		else [_emotions removeObjectForKey:[chat target]];
		return YES;
	}
	return NO;
}

- (void) processMessage:(NSMutableAttributedString *) message asAction:(BOOL) action toRoom:(JVChatRoom *) room {
	if( [_emotions objectForKey:[room target]] ) {
		NSMutableAttributedString *appd = [[[NSMutableAttributedString alloc] initWithString:@" "] autorelease];
		[appd appendAttributedString:[_emotions objectForKey:[room target]]];
		[message appendAttributedString:appd];
	}
}

- (void) processMessage:(NSMutableAttributedString *) message asAction:(BOOL) action toChat:(JVDirectChat *) chat {
	if( [_emotions objectForKey:[chat target]] ) {
		NSMutableAttributedString *appd = [[[NSMutableAttributedString alloc] initWithString:@" "] autorelease];
		[appd appendAttributedString:[_emotions objectForKey:[chat target]]];
		[message appendAttributedString:appd];
	}
}
@end