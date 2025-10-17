/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#include "InspectorResourceURLSchemeHandler.h"

#include <WebCore/File.h>
#include <WebCore/ResourceError.h>
#include <WebCore/WebCoreBundleWin.h>
#include <winsock2.h> // This is required for curl.h
#include <wtf/FileSystem.h>
#include <wtf/URL.h>

#if USE(CURL)
#include <curl/curl.h>
#else
#error Unknown network backend
#endif

namespace WebKit {

void InspectorResourceURLSchemeHandler::platformStartTask(WebPageProxy&, WebURLSchemeTask& task)
{
    auto requestURL = task.request().url();
    auto requestPath = makeStringByReplacingAll(requestURL.path(), '/', '\\');
    if (requestPath.startsWith("\\"_s))
        requestPath = requestPath.substring(1);
    auto path = WebCore::webKitBundlePath({ "WebInspectorUI"_s, requestPath });
    bool success;
    FileSystem::MappedFileData file(path, FileSystem::MappedFileMode::Private, success);
    if (!success) {
        task.didComplete(WebCore::ResourceError(CURLE_READ_ERROR, requestURL));
        return;
    }
    auto contentType = WebCore::File::contentTypeForFile(path);
    if (contentType.isEmpty())
        contentType = "application/octet-stream"_s;
    WebCore::ResourceResponse response(requestURL, contentType, file.size(), "UTF-8"_s);
    auto data = WebCore::SharedBuffer::create(file.span());

    task.didReceiveResponse(response);
    task.didReceiveData(WTFMove(data));
    task.didComplete({ });
}

} // namespace WebKit
