/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#import <WebKitLegacy/DOMCore.h>
#import <WebKitLegacy/DOMDocument.h>
#import <WebKitLegacy/DOMRangeException.h>

@class DOMDocumentFragment;
@class DOMNode;
@class DOMRange;
@class NSString;

enum {
    DOM_START_TO_START = 0,
    DOM_START_TO_END = 1,
    DOM_END_TO_END = 2,
    DOM_END_TO_START = 3,
    DOM_NODE_BEFORE = 0,
    DOM_NODE_AFTER = 1,
    DOM_NODE_BEFORE_AND_AFTER = 2,
    DOM_NODE_INSIDE = 3
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMRange : DOMObject
@property (readonly, strong) DOMNode *startContainer;
@property (readonly) int startOffset;
@property (readonly, strong) DOMNode *endContainer;
@property (readonly) int endOffset;
@property (readonly) BOOL collapsed;
@property (readonly, strong) DOMNode *commonAncestorContainer;
@property (readonly, copy) NSString *text WEBKIT_AVAILABLE_MAC(10_5);

- (void)setStart:(DOMNode *)refNode offset:(int)offset WEBKIT_AVAILABLE_MAC(10_5);
- (void)setEnd:(DOMNode *)refNode offset:(int)offset WEBKIT_AVAILABLE_MAC(10_5);
- (void)setStartBefore:(DOMNode *)refNode;
- (void)setStartAfter:(DOMNode *)refNode;
- (void)setEndBefore:(DOMNode *)refNode;
- (void)setEndAfter:(DOMNode *)refNode;
- (void)collapse:(BOOL)toStart;
- (void)selectNode:(DOMNode *)refNode;
- (void)selectNodeContents:(DOMNode *)refNode;
- (short)compareBoundaryPoints:(unsigned short)how sourceRange:(DOMRange *)sourceRange WEBKIT_AVAILABLE_MAC(10_5);
- (void)deleteContents;
- (DOMDocumentFragment *)extractContents;
- (DOMDocumentFragment *)cloneContents;
- (void)insertNode:(DOMNode *)newNode;
- (void)surroundContents:(DOMNode *)newParent;
- (DOMRange *)cloneRange;
- (NSString *)toString;
- (void)detach;
- (DOMDocumentFragment *)createContextualFragment:(NSString *)html WEBKIT_AVAILABLE_MAC(10_5);
- (short)compareNode:(DOMNode *)refNode WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)intersectsNode:(DOMNode *)refNode WEBKIT_AVAILABLE_MAC(10_5);
- (short)comparePoint:(DOMNode *)refNode offset:(int)offset WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)isPointInRange:(DOMNode *)refNode offset:(int)offset WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMRange (DOMRangeDeprecated)
- (void)setStart:(DOMNode *)refNode :(int)offset WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)setEnd:(DOMNode *)refNode :(int)offset WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (short)compareBoundaryPoints:(unsigned short)how :(DOMRange *)sourceRange WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
