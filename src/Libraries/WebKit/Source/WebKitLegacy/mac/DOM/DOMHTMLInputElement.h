/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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

@class DOMFileList;
@class DOMHTMLFormElement;
@class NSString;
@class NSURL;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLInputElement : DOMHTMLElement
@property (copy) NSString *accept;
@property (copy) NSString *alt;
@property BOOL autofocus WEBKIT_AVAILABLE_MAC(10_6);
@property BOOL defaultChecked;
@property BOOL checked;
@property BOOL disabled;
@property (readonly, strong) DOMHTMLFormElement *form;
@property (strong) DOMFileList *files WEBKIT_AVAILABLE_MAC(10_6);
@property BOOL indeterminate WEBKIT_AVAILABLE_MAC(10_5);
@property int maxLength;
@property BOOL multiple WEBKIT_AVAILABLE_MAC(10_6);
@property (copy) NSString *name;
@property BOOL readOnly;
@property (copy) NSString *size;
@property (copy) NSString *src;
@property (copy) NSString *type;
@property (copy) NSString *defaultValue;
@property (copy) NSString *value;
@property (readonly) BOOL willValidate WEBKIT_AVAILABLE_MAC(10_6);
@property int selectionStart WEBKIT_AVAILABLE_MAC(10_5);
@property int selectionEnd WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *align;
@property (copy) NSString *useMap;
@property (copy) NSString *accessKey WEBKIT_DEPRECATED_MAC(10_4, 10_8);
@property (readonly, copy) NSString *altDisplayString WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSURL *absoluteImageURL WEBKIT_AVAILABLE_MAC(10_5);

- (void)select;
- (void)setSelectionRange:(int)start end:(int)end WEBKIT_AVAILABLE_MAC(10_5);
- (void)click;
@end
