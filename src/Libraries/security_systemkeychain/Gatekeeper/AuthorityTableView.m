/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
//  AuthorityTableView.m
//  security_systemkeychain
//
//  Created by Love HÃ¶rnquist Ã…strand on 2012-03-22.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#import "AuthorityTableView.h"

@implementation AuthorityTableView

- (void)keyDown:(NSEvent *)event
{
    NSString *string = [event charactersIgnoringModifiers];
    if ([string length] == 0) {
	[super keyDown:event];
	return;
    }

    unichar key = [string characterAtIndex:0];
    if (key == NSDeleteCharacter) {
	[self.authorityDelegate deleteAuthority:self atIndexes:[self selectedRowIndexes]];
	return;
    }
    [super keyDown:event];
}

@end
