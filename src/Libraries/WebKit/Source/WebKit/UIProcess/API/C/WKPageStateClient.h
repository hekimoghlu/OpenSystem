/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#pragma once

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageStateClientWillChangeCallback)(const void* clientInfo);
typedef void (*WKPageStateClientDidChangeCallback)(const void* clientInfo);

typedef struct WKPageStateClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKPageStateClientBase;

typedef struct WKPageStateClientV0 {
    WKPageStateClientBase                                               base;

    // Version 0.
    WKPageStateClientWillChangeCallback                                 willChangeIsLoading;
    WKPageStateClientDidChangeCallback                                  didChangeIsLoading;
    WKPageStateClientWillChangeCallback                                 willChangeTitle;
    WKPageStateClientDidChangeCallback                                  didChangeTitle;
    WKPageStateClientWillChangeCallback                                 willChangeActiveURL;
    WKPageStateClientDidChangeCallback                                  didChangeActiveURL;
    WKPageStateClientWillChangeCallback                                 willChangeHasOnlySecureContent;
    WKPageStateClientDidChangeCallback                                  didChangeHasOnlySecureContent;
    WKPageStateClientWillChangeCallback                                 willChangeEstimatedProgress;
    WKPageStateClientDidChangeCallback                                  didChangeEstimatedProgress;
    WKPageStateClientWillChangeCallback                                 willChangeCanGoBack;
    WKPageStateClientDidChangeCallback                                  didChangeCanGoBack;
    WKPageStateClientWillChangeCallback                                 willChangeCanGoForward;
    WKPageStateClientDidChangeCallback                                  didChangeCanGoForward;
    WKPageStateClientWillChangeCallback                                 willChangeNetworkRequestsInProgress;
    WKPageStateClientDidChangeCallback                                  didChangeNetworkRequestsInProgress;
    WKPageStateClientWillChangeCallback                                 willChangeCertificateInfo;
    WKPageStateClientDidChangeCallback                                  didChangeCertificateInfo;
    WKPageStateClientWillChangeCallback                                 willChangeWebProcessIsResponsive;
    WKPageStateClientDidChangeCallback                                  didChangeWebProcessIsResponsive;
    WKPageStateClientDidChangeCallback                                  didSwapWebProcesses;

} WKPageStateClientV0;

#ifdef __cplusplus
}
#endif
