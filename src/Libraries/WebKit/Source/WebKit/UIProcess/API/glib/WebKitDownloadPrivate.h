/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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

#include "APIDownloadClient.h"
#include "DownloadProxy.h"
#include "WebKitDownload.h"
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/WTFString.h>

GRefPtr<WebKitDownload> webkitDownloadCreate(WebKit::DownloadProxy&, WebKitWebView* = nullptr);
void webkitDownloadStarted(WebKitDownload*);
bool webkitDownloadIsCancelled(WebKitDownload*);
void webkitDownloadSetResponse(WebKitDownload*, WebKitURIResponse*);
void webkitDownloadNotifyProgress(WebKitDownload*, guint64 bytesReceived);
void webkitDownloadFailed(WebKitDownload*, const WebCore::ResourceError&);
void webkitDownloadCancelled(WebKitDownload*);
void webkitDownloadFinished(WebKitDownload*);
void webkitDownloadDecideDestinationWithSuggestedFilename(WebKitDownload*, CString&& suggestedFilename, CompletionHandler<void(WebKit::AllowOverwrite, WTF::String)>&&);
void webkitDownloadDestinationCreated(WebKitDownload*, const String& destinationPath);
