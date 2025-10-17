/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef WKContextHistoryClient_h
#define WKContextHistoryClient_h

#include <WebKit/WKBase.h>

typedef void (*WKContextDidNavigateWithNavigationDataCallback)(WKContextRef context, WKPageRef page, WKNavigationDataRef navigationData, WKFrameRef frame, const void *clientInfo);
typedef void (*WKContextDidPerformClientRedirectCallback)(WKContextRef context, WKPageRef page, WKURLRef sourceURL, WKURLRef destinationURL, WKFrameRef frame, const void *clientInfo);
typedef void (*WKContextDidPerformServerRedirectCallback)(WKContextRef context, WKPageRef page, WKURLRef sourceURL, WKURLRef destinationURL, WKFrameRef frame, const void *clientInfo);
typedef void (*WKContextDidUpdateHistoryTitleCallback)(WKContextRef context, WKPageRef page, WKStringRef title, WKURLRef URL, WKFrameRef frame, const void *clientInfo);
typedef void (*WKContextPopulateVisitedLinksCallback)(WKContextRef context, const void *clientInfo);

typedef struct WKContextHistoryClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKContextHistoryClientBase;

typedef struct WKContextHistoryClientV0 {
    WKContextHistoryClientBase                                          base;

    // Version 0.
    WKContextDidNavigateWithNavigationDataCallback                      didNavigateWithNavigationData;
    WKContextDidPerformClientRedirectCallback                           didPerformClientRedirect;
    WKContextDidPerformServerRedirectCallback                           didPerformServerRedirect;
    WKContextDidUpdateHistoryTitleCallback                              didUpdateHistoryTitle;
    WKContextPopulateVisitedLinksCallback                               populateVisitedLinks;
} WKContextHistoryClientV0;

#endif // WKContextHistoryClient_h
