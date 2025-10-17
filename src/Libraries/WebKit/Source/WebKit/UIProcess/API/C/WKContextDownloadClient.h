/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#ifndef WKContextDownloadClient_h
#define WKContextDownloadClient_h

#include <WebKit/WKBase.h>

typedef void (*WKContextDownloadDidStartCallback)(WKContextRef context, WKDownloadRef download, const void *clientInfo);
typedef void (*WKContextDownloadDidReceiveAuthenticationChallengeCallback)(WKContextRef context, WKDownloadRef download, WKAuthenticationChallengeRef authenticationChallenge, const void *clientInfo);
typedef void (*WKContextDownloadDidReceiveResponseCallback)(WKContextRef context, WKDownloadRef download, WKURLResponseRef response, const void *clientInfo);
typedef void (*WKContextDownloadDidReceiveDataCallback)(WKContextRef context, WKDownloadRef download, uint64_t length, const void *clientInfo);
typedef bool (*WKContextDownloadShouldDecodeSourceDataOfMIMETypeCallback)(WKContextRef context, WKDownloadRef download, WKStringRef mimeType, const void *clientInfo);
typedef WKStringRef (*WKContextDownloadDecideDestinationWithSuggestedFilenameCallback)(WKContextRef context, WKDownloadRef download, WKStringRef filename, bool* allowOverwrite, const void *clientInfo);
typedef void (*WKContextDownloadDidCreateDestinationCallback)(WKContextRef context, WKDownloadRef download, WKStringRef path, const void *clientInfo);
typedef void (*WKContextDownloadDidFinishCallback)(WKContextRef context, WKDownloadRef download, const void *clientInfo);
typedef void (*WKContextDownloadDidFailCallback)(WKContextRef context, WKDownloadRef download, WKErrorRef error, const void *clientInfo);
typedef void (*WKContextDownloadDidCancel)(WKContextRef context, WKDownloadRef download, const void *clientInfo);
typedef void (*WKContextDownloadProcessDidCrashCallback)(WKContextRef context, WKDownloadRef download, const void *clientInfo);
typedef void (*WKContextDownloadDidReceiveServerRedirect)(WKContextRef context, WKDownloadRef download, WKURLRef url, const void *clientInfo);

typedef struct WKContextDownloadClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKContextDownloadClientBase;

typedef struct WKContextDownloadClientV0 {
    WKContextDownloadClientBase                                         base;

    // Version 0.
    WKContextDownloadDidStartCallback                                   didStart;
    WKContextDownloadDidReceiveAuthenticationChallengeCallback          didReceiveAuthenticationChallenge;
    WKContextDownloadDidReceiveResponseCallback                         didReceiveResponse;
    WKContextDownloadDidReceiveDataCallback                             didReceiveData;
    WKContextDownloadShouldDecodeSourceDataOfMIMETypeCallback           shouldDecodeSourceDataOfMIMEType;
    WKContextDownloadDecideDestinationWithSuggestedFilenameCallback     decideDestinationWithSuggestedFilename;
    WKContextDownloadDidCreateDestinationCallback                       didCreateDestination;
    WKContextDownloadDidFinishCallback                                  didFinish;
    WKContextDownloadDidFailCallback                                    didFail;
    WKContextDownloadDidCancel                                          didCancel;
    WKContextDownloadProcessDidCrashCallback                            processDidCrash;
} WKContextDownloadClientV0;

typedef struct WKContextDownloadClientV1 {
    WKContextDownloadClientBase                                         base;

    // Version 0.
    WKContextDownloadDidStartCallback                                   didStart;
    WKContextDownloadDidReceiveAuthenticationChallengeCallback          didReceiveAuthenticationChallenge;
    WKContextDownloadDidReceiveResponseCallback                         didReceiveResponse;
    WKContextDownloadDidReceiveDataCallback                             didReceiveData;
    WKContextDownloadShouldDecodeSourceDataOfMIMETypeCallback           shouldDecodeSourceDataOfMIMEType;
    WKContextDownloadDecideDestinationWithSuggestedFilenameCallback     decideDestinationWithSuggestedFilename;
    WKContextDownloadDidCreateDestinationCallback                       didCreateDestination;
    WKContextDownloadDidFinishCallback                                  didFinish;
    WKContextDownloadDidFailCallback                                    didFail;
    WKContextDownloadDidCancel                                          didCancel;
    WKContextDownloadProcessDidCrashCallback                            processDidCrash;

    // Version 1.
    WKContextDownloadDidReceiveServerRedirect                           didReceiveServerRedirect;
} WKContextDownloadClientV1;

#endif // WKContextDownloadClient_h
