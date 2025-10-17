/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

@class DOMNode;
@class NSString;

enum {
    DOM_ANY_TYPE = 0,
    DOM_NUMBER_TYPE = 1,
    DOM_STRING_TYPE = 2,
    DOM_BOOLEAN_TYPE = 3,
    DOM_UNORDERED_NODE_ITERATOR_TYPE = 4,
    DOM_ORDERED_NODE_ITERATOR_TYPE = 5,
    DOM_UNORDERED_NODE_SNAPSHOT_TYPE = 6,
    DOM_ORDERED_NODE_SNAPSHOT_TYPE = 7,
    DOM_ANY_UNORDERED_NODE_TYPE = 8,
    DOM_FIRST_ORDERED_NODE_TYPE = 9
} WEBKIT_ENUM_DEPRECATED_MAC(10_5, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_5, 10_14)
@interface DOMXPathResult : DOMObject
@property (readonly) unsigned short resultType;
@property (readonly) double numberValue;
@property (readonly, copy) NSString *stringValue;
@property (readonly) BOOL booleanValue;
@property (readonly, strong) DOMNode *singleNodeValue;
@property (readonly) BOOL invalidIteratorState;
@property (readonly) unsigned snapshotLength;

- (DOMNode *)iterateNext;
- (DOMNode *)snapshotItem:(unsigned)index;
@end
