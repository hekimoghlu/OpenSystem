/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "WebFrameNetworkingContext.h"

#include "MessageSenderInlines.h"
#include "NetworkSession.h"
#include "NetworkSessionCreationParameters.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebsiteDataStoreParameters.h"
#include <WebCore/FrameLoader.h>
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/Settings.h>
#include <WebCore/SoupNetworkSession.h>

namespace WebKit {
using namespace WebCore;

void WebFrameNetworkingContext::ensureWebsiteDataStoreSession(const WebsiteDataStoreParameters&)
{
}

WebFrameNetworkingContext::WebFrameNetworkingContext(WebFrame* frame)
    : FrameNetworkingContext(frame->coreLocalFrame())
{
}

WebLocalFrameLoaderClient* WebFrameNetworkingContext::webFrameLoaderClient() const
{
    if (!frame())
        return nullptr;

    return toWebLocalFrameLoaderClient(frame()->loader().client());
}

}
