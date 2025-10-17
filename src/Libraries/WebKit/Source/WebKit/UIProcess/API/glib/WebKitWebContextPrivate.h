/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "WebKitUserContentManager.h"
#include "WebKitWebContext.h"
#include "WebKitWebsitePolicies.h"
#include "WebProcessPool.h"
#include <WebCore/ResourceRequest.h>

WebKit::WebProcessPool& webkitWebContextGetProcessPool(WebKitWebContext*);
#if !ENABLE(2022_GLIB_API)
void webkitWebContextDownloadStarted(WebKitWebContext*, WebKitDownload*);
#endif
void webkitWebContextWebViewCreated(WebKitWebContext*, WebKitWebView*);
void webkitWebContextWebViewDestroyed(WebKitWebContext*, WebKitWebView*);
WebKitWebView* webkitWebContextGetWebViewForPage(WebKitWebContext*, WebKit::WebPageProxy*);
GVariant* webkitWebContextInitializeWebProcessExtensions(WebKitWebContext*);
void webkitWebContextInitializeNotificationPermissions(WebKitWebContext*);
#if ENABLE(REMOTE_INSPECTOR)
void webkitWebContextWillCloseAutomationSession(WebKitWebContext*);
#endif
