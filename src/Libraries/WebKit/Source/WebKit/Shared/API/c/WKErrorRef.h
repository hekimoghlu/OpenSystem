/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#ifndef WKErrorRef_h
#define WKErrorRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKErrorCodeCannotShowMIMEType =                             100,
    kWKErrorCodeCannotShowURL =                                  101,
    kWKErrorCodeFrameLoadInterruptedByPolicyChange =             102,
    kWKErrorCodeCannotUseRestrictedPort =                        103,
    kWKErrorCodeFrameLoadBlockedByContentBlocker =               104,
    kWKErrorCodeFrameLoadBlockedByContentFilter =                105,
    kWKErrorCodeFrameLoadBlockedByRestrictions =                 106,
    kWKErrorCodeCannotFindPlugIn =                               200,
    kWKErrorCodeCannotLoadPlugIn =                               201,
    kWKErrorCodeJavaUnavailable =                                202,
    kWKErrorCodePlugInCancelledConnection =                      203,
    kWKErrorCodePlugInWillHandleLoad =                           204,
    kWKErrorCodeInsecurePlugInVersion =                          205,
    kWKErrorInternal =                                           300,
    kWKErrorCodeCancelled =                                      302,
    kWKErrorCodeFileDoesNotExist =                               303,
};

WK_EXPORT WKTypeID WKErrorGetTypeID(void);

WK_EXPORT WKStringRef WKErrorCopyWKErrorDomain(void);

WK_EXPORT WKStringRef WKErrorCopyDomain(WKErrorRef error);
WK_EXPORT int WKErrorGetErrorCode(WKErrorRef error);
WK_EXPORT WKURLRef WKErrorCopyFailingURL(WKErrorRef error);
WK_EXPORT WKStringRef WKErrorCopyLocalizedDescription(WKErrorRef error);

#ifdef __cplusplus
}
#endif

#endif // WKErrorRef_h
