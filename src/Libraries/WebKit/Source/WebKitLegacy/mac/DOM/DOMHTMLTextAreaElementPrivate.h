/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#import <WebKitLegacy/DOMHTMLElementPrivate.h>
#import <WebKitLegacy/DOMHTMLTextAreaElement.h>

@class DOMNodeList;
@class DOMValidityState;

@interface DOMHTMLTextAreaElement (DOMHTMLTextAreaElementPrivate)
@property (copy) NSString *dirName;
@property int maxLength;
@property (copy) NSString *placeholder;
@property BOOL required;
@property (copy) NSString *wrap;
@property (readonly) unsigned textLength;
@property (readonly, strong) DOMNodeList *labels;
@property (copy) NSString *selectionDirection;
@property (copy) NSString *autocomplete;
@property (nonatomic) BOOL canShowPlaceholder;
- (void)setRangeText:(NSString *)replacement;
- (void)setRangeText:(NSString *)replacement start:(unsigned)start end:(unsigned)end selectionMode:(NSString *)selectionMode;
@end
