/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#import <WebKitLegacy/DOMHTMLElement.h>

@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLTableCellElement : DOMHTMLElement
@property (readonly) int cellIndex;
@property (copy) NSString *align;
@property (copy) NSString *axis;
@property (copy) NSString *bgColor;
@property (copy) NSString *ch;
@property (copy) NSString *chOff;
@property int colSpan;
@property int rowSpan;
@property (copy) NSString *headers;
@property (copy) NSString *height;
@property BOOL noWrap;
@property (copy) NSString *vAlign;
@property (copy) NSString *width;
@property (copy) NSString *abbr;
@property (copy) NSString *scope;
@end
