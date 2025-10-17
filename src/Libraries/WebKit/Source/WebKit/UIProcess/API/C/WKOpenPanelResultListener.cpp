/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "config.h"
#include "WKOpenPanelResultListener.h"

#include "APIArray.h"
#include "APIData.h"
#include "APIString.h"
#include "WKAPICast.h"
#include "WebOpenPanelResultListenerProxy.h"
#include <wtf/URL.h>

using namespace WebKit;

WKTypeID WKOpenPanelResultListenerGetTypeID()
{
    return toAPI(WebOpenPanelResultListenerProxy::APIType);
}

static Vector<String> filePathsFromFileURLs(const API::Array& fileURLs)
{
    Vector<String> filePaths;

    size_t size = fileURLs.size();
    filePaths.reserveInitialCapacity(size);

    for (size_t i = 0; i < size; ++i) {
        RefPtr apiURL = fileURLs.at<API::URL>(i);
        if (apiURL)
            filePaths.append(URL { apiURL->string() }.fileSystemPath());
    }

    return filePaths;
}

#if PLATFORM(IOS_FAMILY)
void WKOpenPanelResultListenerChooseMediaFiles(WKOpenPanelResultListenerRef listenerRef, WKArrayRef fileURLsRef, WKStringRef displayString, WKDataRef iconImageDataRef)
{
    toImpl(listenerRef)->chooseFiles(filePathsFromFileURLs(*toImpl(fileURLsRef)), toImpl(displayString)->string(), toImpl(iconImageDataRef));
}
#endif

void WKOpenPanelResultListenerChooseFiles(WKOpenPanelResultListenerRef listenerRef, WKArrayRef fileURLsRef, WKArrayRef allowedMimeTypesRef)
{
    toImpl(listenerRef)->chooseFiles(filePathsFromFileURLs(*toImpl(fileURLsRef)), toImpl(allowedMimeTypesRef)->toStringVector());
}

void WKOpenPanelResultListenerCancel(WKOpenPanelResultListenerRef listenerRef)
{
    toImpl(listenerRef)->cancel();
}
