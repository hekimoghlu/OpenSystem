/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef WKPageDiagnosticLoggingClient_h
#define WKPageDiagnosticLoggingClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDiagnosticLoggingResultType.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageLogDiagnosticMessageCallback)(WKPageRef page, WKStringRef message, WKStringRef description, const void* clientInfo);
typedef void (*WKPageLogDiagnosticMessageWithResultCallback)(WKPageRef page, WKStringRef message, WKStringRef description, WKDiagnosticLoggingResultType result, const void* clientInfo);
typedef void (*WKPageLogDiagnosticMessageWithValueCallback)(WKPageRef page, WKStringRef message, WKStringRef description, WKStringRef value, const void* clientInfo);
typedef void (*WKPageLogDiagnosticMessageWithEnhancedPrivacyCallback)(WKPageRef page, WKStringRef message, WKStringRef description, const void* clientInfo);

typedef struct WKPageDiagnosticLoggingClientBase {
    int                                                                version;
    const void *                                                       clientInfo;
} WKPageDiagnosticLoggingClientBase;

typedef struct WKPageDiagnosticLoggingClientV0 {
    WKPageDiagnosticLoggingClientBase                                  base;

    // Version 0.
    WKPageLogDiagnosticMessageCallback                                 logDiagnosticMessage;
    WKPageLogDiagnosticMessageWithResultCallback                       logDiagnosticMessageWithResult;
    WKPageLogDiagnosticMessageWithValueCallback                        logDiagnosticMessageWithValue;
} WKPageDiagnosticLoggingClientV0;

typedef struct WKPageDiagnosticLoggingClientV1 {
    WKPageDiagnosticLoggingClientBase                                  base;

    // Version 0.
    WKPageLogDiagnosticMessageCallback                                 logDiagnosticMessage;
    WKPageLogDiagnosticMessageWithResultCallback                       logDiagnosticMessageWithResult;
    WKPageLogDiagnosticMessageWithValueCallback                        logDiagnosticMessageWithValue;

    // Version 1.
    WKPageLogDiagnosticMessageWithEnhancedPrivacyCallback              logDiagnosticMessageWithEnhancedPrivacy;
} WKPageDiagnosticLoggingClientV1;

#ifdef __cplusplus
}
#endif

#endif // WKPageDiagnosticLoggingClient_h
