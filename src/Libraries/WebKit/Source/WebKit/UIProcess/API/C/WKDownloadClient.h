/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#ifndef WKDownloadClient_h
#define WKDownloadClient_h

#include <WebKit/WKBase.h>

typedef bool (*WKDownloadWillPerformHTTPRedirection)(WKDownloadRef download, WKURLResponseRef response, WKURLRequestRef newRequest, const void* clientInfo);
typedef void (*WKDownloadDidReceiveAuthenticationChallenge)(WKDownloadRef download, WKAuthenticationChallengeRef challenge, const void* clientInfo);
typedef WKStringRef (*WKDownloadDecideDestinationWithResponse)(WKDownloadRef download, WKURLResponseRef response, WKStringRef suggestedFilename, const void* clientInfo);
typedef void (*WKDownloadDidWriteData)(WKDownloadRef download, long long bytesWritten, long long totalBytesWritten, long long totalBytesExpectedToWrite, const void* clientInfo);
typedef void (*WKDownloadDidFinish)(WKDownloadRef download, const void* clientInfo);
typedef void (*WKDownloadDidFailWithError)(WKDownloadRef download, WKErrorRef error, WKDataRef resumeData, const void* clientInfo);

typedef struct WKDownloadClientBase {
    int                                                        version;
    const void *                                               clientInfo;
} WKDownloadClientBase;

typedef struct WKDownloadClientV0 {
    WKDownloadClientBase                                       base;

    // Version 0.
    WKDownloadWillPerformHTTPRedirection                       willPerformHTTPRedirection;
    WKDownloadDidReceiveAuthenticationChallenge                didReceiveAuthenticationChallenge;
    WKDownloadDecideDestinationWithResponse                    decideDestinationWithResponse;
    WKDownloadDidWriteData                                     didWriteData;
    WKDownloadDidFinish                                        didFinish;
    WKDownloadDidFailWithError                                 didFailWithError;
} WKDownloadClientV0;

#endif /* WKDownloadClient_h */
