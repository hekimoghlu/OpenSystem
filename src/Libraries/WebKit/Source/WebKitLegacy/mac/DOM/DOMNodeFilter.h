/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
#import <WebKitLegacy/WebKitAvailability.h>
#import <objc/NSObject.h>

@class DOMNode;

enum {
    DOM_FILTER_ACCEPT = 1,
    DOM_FILTER_REJECT = 2,
    DOM_FILTER_SKIP = 3,
    DOM_SHOW_ALL = 0xFFFFFFFF,
    DOM_SHOW_ELEMENT = 0x00000001,
    DOM_SHOW_ATTRIBUTE = 0x00000002,
    DOM_SHOW_TEXT = 0x00000004,
    DOM_SHOW_CDATA_SECTION = 0x00000008,
    DOM_SHOW_ENTITY_REFERENCE = 0x00000010,
    DOM_SHOW_ENTITY = 0x00000020,
    DOM_SHOW_PROCESSING_INSTRUCTION = 0x00000040,
    DOM_SHOW_COMMENT = 0x00000080,
    DOM_SHOW_DOCUMENT = 0x00000100,
    DOM_SHOW_DOCUMENT_TYPE = 0x00000200,
    DOM_SHOW_DOCUMENT_FRAGMENT = 0x00000400,
    DOM_SHOW_NOTATION = 0x00000800
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@protocol DOMNodeFilter <NSObject>
- (short)acceptNode:(DOMNode *)n;
@end
