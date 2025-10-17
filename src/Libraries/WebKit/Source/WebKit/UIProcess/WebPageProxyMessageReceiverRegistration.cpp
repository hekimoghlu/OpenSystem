/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "WebPageProxyMessageReceiverRegistration.h"

#include "WebPageProxyMessages.h"
#include "WebProcessProxy.h"

namespace WebKit {

WebPageProxyMessageReceiverRegistration::~WebPageProxyMessageReceiverRegistration()
{
    stopReceivingMessages();
}

void WebPageProxyMessageReceiverRegistration::startReceivingMessages(WebProcessProxy& process, WebCore::PageIdentifier webPageID, IPC::MessageReceiver& messageReceiver)
{
    stopReceivingMessages();
    process.addMessageReceiver(Messages::WebPageProxy::messageReceiverName(), webPageID, messageReceiver);
    m_data = { { webPageID, process } };
}

void WebPageProxyMessageReceiverRegistration::stopReceivingMessages()
{
    if (auto data = std::exchange(m_data, std::nullopt))
        data->protectedProcess()->removeMessageReceiver(Messages::WebPageProxy::messageReceiverName(), data->webPageID);
}

void WebPageProxyMessageReceiverRegistration::transferMessageReceivingFrom(WebPageProxyMessageReceiverRegistration& oldRegistration, IPC::MessageReceiver& newReceiver)
{
    ASSERT(!m_data);
    if (auto data = std::exchange(oldRegistration.m_data, std::nullopt)) {
        data->protectedProcess()->removeMessageReceiver(Messages::WebPageProxy::messageReceiverName(), data->webPageID);
        startReceivingMessages(data->process, data->webPageID, newReceiver);
    } else {
        stopReceivingMessages();
        ASSERT_NOT_REACHED();
    }
}

Ref<WebProcessProxy> WebPageProxyMessageReceiverRegistration::Data::protectedProcess()
{
    return process;
}

} // namespace WebKit
