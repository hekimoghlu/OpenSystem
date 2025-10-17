/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

@class DOMNode;
@class NSString;

enum {
    DOM_MODIFICATION = 1,
    DOM_ADDITION = 2,
    DOM_REMOVAL = 3
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMMutationEvent : DOMEvent
@property (readonly, strong) DOMNode *relatedNode;
@property (readonly, copy) NSString *prevValue;
@property (readonly, copy) NSString *newValue;
- (NSString *)newValue NS_RETURNS_NOT_RETAINED;
@property (readonly, copy) NSString *attrName;
@property (readonly) unsigned short attrChange;

- (void)initMutationEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable relatedNode:(DOMNode *)relatedNode prevValue:(NSString *)prevValue newValue:(NSString *)newValue attrName:(NSString *)attrName attrChange:(unsigned short)attrChange WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMMutationEvent (DOMMutationEventDeprecated)
- (void)initMutationEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMNode *)relatedNode :(NSString *)prevValue :(NSString *)newValue :(NSString *)attrName :(unsigned short)attrChange WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
