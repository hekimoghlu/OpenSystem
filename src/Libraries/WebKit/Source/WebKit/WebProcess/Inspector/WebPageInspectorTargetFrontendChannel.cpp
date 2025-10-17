/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#include "WebPageInspectorTargetFrontendChannel.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageInspectorTargetFrontendChannel);

WebPageInspectorTargetFrontendChannel::WebPageInspectorTargetFrontendChannel(WebPage& page, const String& targetId, Inspector::FrontendChannel::ConnectionType connectionType)
    : m_page(page)
    , m_targetId(targetId)
    , m_connectionType(connectionType)
{
}

void WebPageInspectorTargetFrontendChannel::sendMessageToFrontend(const String& message)
{
    Ref { m_page.get() }->send(Messages::WebPageProxy::SendMessageToInspectorFrontend(m_targetId, message));
}

} // namespace WebKit
