/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

@class NSString;

extern NSString * const DOMException WEBKIT_DEPRECATED_MAC(10_4, 10_14);

enum DOMExceptionCode {
    DOM_INDEX_SIZE_ERR                = 1,
    DOM_DOMSTRING_SIZE_ERR            = 2,
    DOM_HIERARCHY_REQUEST_ERR         = 3,
    DOM_WRONG_DOCUMENT_ERR            = 4,
    DOM_INVALID_CHARACTER_ERR         = 5,
    DOM_NO_DATA_ALLOWED_ERR           = 6,
    DOM_NO_MODIFICATION_ALLOWED_ERR   = 7,
    DOM_NOT_FOUND_ERR                 = 8,
    DOM_NOT_SUPPORTED_ERR             = 9,
    DOM_INUSE_ATTRIBUTE_ERR           = 10,
    DOM_INVALID_STATE_ERR             = 11,
    DOM_SYNTAX_ERR                    = 12,
    DOM_INVALID_MODIFICATION_ERR      = 13,
    DOM_NAMESPACE_ERR                 = 14,
    DOM_INVALID_ACCESS_ERR            = 15
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);
