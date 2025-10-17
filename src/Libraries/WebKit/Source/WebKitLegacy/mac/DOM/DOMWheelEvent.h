/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#import <WebKitLegacy/DOMMouseEvent.h>

@class DOMAbstractView;

enum {
    DOM_DOM_DELTA_PIXEL = 0x00,
    DOM_DOM_DELTA_LINE = 0x01,
    DOM_DOM_DELTA_PAGE = 0x02
} WEBKIT_ENUM_DEPRECATED_MAC(10_5, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_5, 10_14)
@interface DOMWheelEvent : DOMMouseEvent
@property (readonly) int wheelDeltaX WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int wheelDeltaY WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int wheelDelta;
@property (readonly) BOOL isHorizontal;

- (void)initWheelEvent:(int)wheelDeltaX wheelDeltaY:(int)wheelDeltaY view:(DOMAbstractView *)view screenX:(int)screenX screenY:(int)screenY clientX:(int)clientX clientY:(int)clientY ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey WEBKIT_AVAILABLE_MAC(10_5);
@end
