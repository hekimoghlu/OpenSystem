/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#import <WebKitLegacy/DOMObject.h>

@class NSString;
@protocol DOMEventTarget;

enum {
    DOM_NONE = 0,
    DOM_CAPTURING_PHASE = 1,
    DOM_AT_TARGET = 2,
    DOM_BUBBLING_PHASE = 3
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMEvent : DOMObject
@property (readonly, copy) NSString *type;
@property (readonly, strong) id <DOMEventTarget> target;
@property (readonly, strong) id <DOMEventTarget> currentTarget;
@property (readonly) unsigned short eventPhase;
@property (readonly) BOOL bubbles;
@property (readonly) BOOL cancelable;
@property (readonly) DOMTimeStamp timeStamp;
@property (readonly, strong) id <DOMEventTarget> srcElement WEBKIT_AVAILABLE_MAC(10_6);
@property BOOL returnValue WEBKIT_AVAILABLE_MAC(10_6);
@property BOOL cancelBubble WEBKIT_AVAILABLE_MAC(10_6);

- (void)stopPropagation;
- (void)preventDefault;
- (void)initEvent:(NSString *)eventTypeArg canBubbleArg:(BOOL)canBubbleArg cancelableArg:(BOOL)cancelableArg WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMEvent (DOMEventDeprecated)
- (void)initEvent:(NSString *)eventTypeArg :(BOOL)canBubbleArg :(BOOL)cancelableArg WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
