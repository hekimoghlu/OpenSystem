/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include <WebCore/PageIdentifier.h>

namespace IPC {
class MessageReceiver;
}

namespace WebKit {

class WebProcessProxy;

class WebPageProxyMessageReceiverRegistration {
public:
    ~WebPageProxyMessageReceiverRegistration();
    void startReceivingMessages(WebProcessProxy&, WebCore::PageIdentifier, IPC::MessageReceiver&);
    void stopReceivingMessages();
    void transferMessageReceivingFrom(WebPageProxyMessageReceiverRegistration&, IPC::MessageReceiver& newReceiver);
private:
    struct Data {
        WebCore::PageIdentifier webPageID;
        Ref<WebProcessProxy> process;

        Ref<WebProcessProxy> protectedProcess();
    };
    std::optional<Data> m_data;
};

} // namespace WebKit
