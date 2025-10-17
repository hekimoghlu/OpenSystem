/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "WebSleepDisablerClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSleepDisablerClient);

void WebSleepDisablerClient::didCreateSleepDisabler(WebCore::SleepDisablerIdentifier identifier, const String& reason, bool display, std::optional<WebCore::PageIdentifier> pageID)
{
    if (auto* webPage = pageID ? WebProcess::singleton().webPage(*pageID) : nullptr)
        webPage->send(Messages::WebPageProxy::DidCreateSleepDisabler(identifier, reason, display));
}

void WebSleepDisablerClient::didDestroySleepDisabler(WebCore::SleepDisablerIdentifier identifier, std::optional<WebCore::PageIdentifier> pageID)
{
    if (auto* webPage = pageID ? WebProcess::singleton().webPage(*pageID) : nullptr)
        webPage->send(Messages::WebPageProxy::DidDestroySleepDisabler(identifier));
}

} // namespace WebKit
