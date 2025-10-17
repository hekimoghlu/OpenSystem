/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

@class DOMHTMLFormElement;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLTextAreaElement : DOMHTMLElement
@property BOOL autofocus WEBKIT_AVAILABLE_MAC(10_6);
@property BOOL disabled;
@property (readonly, strong) DOMHTMLFormElement *form;
@property (copy) NSString *name;
@property BOOL readOnly;
@property int rows;
@property int cols;
@property (readonly, copy) NSString *type;
@property (copy) NSString *defaultValue;
@property (copy) NSString *value;
@property (readonly) BOOL willValidate WEBKIT_AVAILABLE_MAC(10_6);
@property int selectionStart WEBKIT_AVAILABLE_MAC(10_5);
@property int selectionEnd WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *accessKey WEBKIT_DEPRECATED_MAC(10_4, 10_8);
- (void)select;
- (void)setSelectionRange:(int)start end:(int)end WEBKIT_AVAILABLE_MAC(10_5);
@end
