/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

@class DOMHTMLCollection;
@class DOMHTMLElement;
@class DOMHTMLTableCaptionElement;
@class DOMHTMLTableSectionElement;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLTableElement : DOMHTMLElement
@property (strong) DOMHTMLTableCaptionElement *caption;
@property (strong) DOMHTMLTableSectionElement *tHead;
@property (strong) DOMHTMLTableSectionElement *tFoot;
@property (readonly, strong) DOMHTMLCollection *rows;
@property (readonly, strong) DOMHTMLCollection *tBodies;
@property (copy) NSString *align;
@property (copy) NSString *bgColor;
@property (copy) NSString *border;
@property (copy) NSString *cellPadding;
@property (copy) NSString *cellSpacing;
@property (copy) NSString *frameBorders;
@property (copy) NSString *rules;
@property (copy) NSString *summary;
@property (copy) NSString *width;

- (DOMHTMLElement *)createTHead;
- (void)deleteTHead;
- (DOMHTMLElement *)createTFoot;
- (void)deleteTFoot;
- (DOMHTMLElement *)createCaption;
- (void)deleteCaption;
- (DOMHTMLElement *)insertRow:(int)index;
- (void)deleteRow:(int)index;
@end
