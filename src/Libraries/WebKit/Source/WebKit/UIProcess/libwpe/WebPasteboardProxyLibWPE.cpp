/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "WebPasteboardProxy.h"

#include <WebCore/PlatformPasteboard.h>
#include <wtf/CompletionHandler.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

void WebPasteboardProxy::getPasteboardTypes(CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    Vector<String> pasteboardTypes;
    PlatformPasteboard().getTypes(pasteboardTypes);
    completionHandler(WTFMove(pasteboardTypes));
}

void WebPasteboardProxy::readStringFromPasteboard(IPC::Connection&, size_t index, const String& pasteboardType, const String&, std::optional<PageIdentifier>, CompletionHandler<void(String&&)>&& completionHandler)
{
    completionHandler(PlatformPasteboard().readString(index, pasteboardType));
}

void WebPasteboardProxy::writeWebContentToPasteboard(const WebCore::PasteboardWebContent& content)
{
    PlatformPasteboard().write(content);
}

void WebPasteboardProxy::writeStringToPasteboard(const String& pasteboardType, const String& text)
{
    PlatformPasteboard().write(pasteboardType, text);
}

} // namespace WebKit
