/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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

extern NSString *WebKitErrorDomain WEBKIT_DEPRECATED_MAC(10_3, 10_14);

extern NSString * const WebKitErrorMIMETypeKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);
extern NSString * const WebKitErrorPlugInNameKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);
extern NSString * const WebKitErrorPlugInPageURLStringKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @enum
    @abstract Policy errors
    @constant WebKitErrorCannotShowMIMEType
    @constant WebKitErrorCannotShowURL
    @constant WebKitErrorFrameLoadInterruptedByPolicyChange
*/
enum {
    WebKitErrorCannotShowMIMEType =                             100,
    WebKitErrorCannotShowURL =                                  101,
    WebKitErrorFrameLoadInterruptedByPolicyChange =             102,
} WEBKIT_ENUM_DEPRECATED_MAC(10_3, 10_14);

/*!
    @enum
    @abstract Plug-in and java errors
    @constant WebKitErrorCannotFindPlugIn
    @constant WebKitErrorCannotLoadPlugIn
    @constant WebKitErrorJavaUnavailable
    @constant WebKitErrorBlockedPlugInVersion
*/
enum {
    WebKitErrorCannotFindPlugIn =                               200,
    WebKitErrorCannotLoadPlugIn =                               201,
    WebKitErrorJavaUnavailable =                                202,
    WebKitErrorBlockedPlugInVersion =                           203,
} WEBKIT_ENUM_DEPRECATED_MAC(10_3, 10_14);
