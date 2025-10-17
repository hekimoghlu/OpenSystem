/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#ifndef WAKResponder_h
#define WAKResponder_h

#import <Foundation/Foundation.h>

#if TARGET_OS_IPHONE

#import <WebCore/WKTypes.h>

@class WebEvent;

WEBCORE_EXPORT @interface WAKResponder : NSObject
{

}

- (void)handleEvent:(WebEvent *)event;

- (void)scrollWheel:(WebEvent *)theEvent;
- (BOOL)tryToPerform:(SEL)anAction with:(id)anObject;
- (void)mouseEntered:(WebEvent *)theEvent;
- (void)mouseExited:(WebEvent *)theEvent;
- (void)keyDown:(WebEvent *)event;
- (void)keyUp:(WebEvent *)event;
#if defined(ENABLE_TOUCH_EVENTS) && ENABLE_TOUCH_EVENTS
- (void)touch:(WebEvent *)event;
#endif

- (void)insertText:(NSString *)text;

- (void)deleteBackward:(id)sender;
- (void)deleteForward:(id)sender;
- (void)insertParagraphSeparator:(id)sender;

- (void)moveDown:(id)sender;
- (void)moveDownAndModifySelection:(id)sender;
- (void)moveLeft:(id)sender;
- (void)moveLeftAndModifySelection:(id)sender;
- (void)moveRight:(id)sender;
- (void)moveRightAndModifySelection:(id)sender;
- (void)moveUp:(id)sender;
- (void)moveUpAndModifySelection:(id)sender;

- (WAKResponder *)nextResponder;
- (BOOL)acceptsFirstResponder;
- (BOOL)becomeFirstResponder;
- (BOOL)resignFirstResponder;

- (void)mouseUp:(WebEvent *)theEvent;
- (void)mouseDown:(WebEvent *)theEvent;
- (void)mouseMoved:(WebEvent *)theEvent;

@end

#endif // TARGET_OS_IPHONE

#endif // WAKResponder_h
