/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#import <WebKitLegacy/DOMEvent.h>

@class DOMAbstractView;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMUIEvent : DOMEvent
@property (readonly, strong) DOMAbstractView *view;
@property (readonly) int detail;
@property (readonly) int keyCode WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int charCode WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int layerX WEBKIT_DEPRECATED_MAC(10_5, 10_5);
@property (readonly) int layerY WEBKIT_DEPRECATED_MAC(10_5, 10_5);
@property (readonly) int pageX WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int pageY WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int which WEBKIT_AVAILABLE_MAC(10_5);

- (void)initUIEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view detail:(int)detail WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMUIEvent (DOMUIEventDeprecated)
- (void)initUIEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMAbstractView *)view :(int)detail WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
