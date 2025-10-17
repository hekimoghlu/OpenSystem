/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
@class NSString;

enum {
    DOM_KEY_LOCATION_STANDARD = 0x00,
    DOM_KEY_LOCATION_LEFT = 0x01,
    DOM_KEY_LOCATION_RIGHT = 0x02,
    DOM_KEY_LOCATION_NUMPAD = 0x03
} WEBKIT_ENUM_DEPRECATED_MAC(10_5, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_5, 10_14)
@interface DOMKeyboardEvent : DOMUIEvent
@property (readonly, copy) NSString *keyIdentifier;
@property (readonly) unsigned location WEBKIT_AVAILABLE_MAC(10_8);
@property (readonly) unsigned keyLocation WEBKIT_DEPRECATED_MAC(10_5, 10_5);
@property (readonly) BOOL ctrlKey;
@property (readonly) BOOL shiftKey;
@property (readonly) BOOL altKey;
@property (readonly) BOOL metaKey;
@property (readonly) BOOL altGraphKey WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int keyCode;
@property (readonly) int charCode;

- (BOOL)getModifierState:(NSString *)keyIdentifierArg;
- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)keyIdentifier location:(unsigned)location ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey altGraphKey:(BOOL)altGraphKey WEBKIT_AVAILABLE_MAC(10_8);
- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)keyIdentifier location:(unsigned)location ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey WEBKIT_AVAILABLE_MAC(10_8);
- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)keyIdentifier keyLocation:(unsigned)keyLocation ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey altGraphKey:(BOOL)altGraphKey WEBKIT_DEPRECATED_MAC(10_5, 10_5);
- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)keyIdentifier keyLocation:(unsigned)keyLocation ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey WEBKIT_DEPRECATED_MAC(10_5, 10_5);
@end
