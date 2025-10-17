/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#import <WebKitLegacy/DOMUIEvent.h>

@class DOMAbstractView;
@class DOMNode;
@class NSString;
@protocol DOMEventTarget;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMMouseEvent : DOMUIEvent
@property (readonly) int screenX;
@property (readonly) int screenY;
@property (readonly) int clientX;
@property (readonly) int clientY;
@property (readonly) BOOL ctrlKey;
@property (readonly) BOOL shiftKey;
@property (readonly) BOOL altKey;
@property (readonly) BOOL metaKey;
@property (readonly) short button;
@property (readonly, strong) id <DOMEventTarget> relatedTarget;
@property (readonly) int offsetX WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int offsetY WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int x WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int y WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMNode *fromElement WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMNode *toElement WEBKIT_AVAILABLE_MAC(10_5);

- (void)initMouseEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view detail:(int)detail screenX:(int)screenX screenY:(int)screenY clientX:(int)clientX clientY:(int)clientY ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey button:(unsigned short)button relatedTarget:(id <DOMEventTarget>)relatedTarget WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMMouseEvent (DOMMouseEventDeprecated)
- (void)initMouseEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMAbstractView *)view :(int)detail :(int)screenX :(int)screenY :(int)clientX :(int)clientY :(BOOL)ctrlKey :(BOOL)altKey :(BOOL)shiftKey :(BOOL)metaKey :(unsigned short)button :(id <DOMEventTarget>)relatedTarget WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
