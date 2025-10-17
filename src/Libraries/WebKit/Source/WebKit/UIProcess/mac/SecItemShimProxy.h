/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

#if ENABLE(SEC_ITEM_SHIM)

#include "Connection.h"
#include "WorkQueueMessageReceiver.h"

namespace WebKit {

class SecItemRequestData;
class SecItemResponseData;

class SecItemShimProxy final : private IPC::MessageReceiver {
WTF_MAKE_NONCOPYABLE(SecItemShimProxy);
public:
    static SecItemShimProxy& singleton();

    void initializeConnection(IPC::Connection&);

    // Do nothing since this is a singleton.
    void ref() const final { }
    void deref() const final { }

private:
    SecItemShimProxy();
    ~SecItemShimProxy();

    // IPC::Connection::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) override;

    void secItemRequest(IPC::Connection&, const SecItemRequestData&, CompletionHandler<void(std::optional<SecItemResponseData>&&)>&&);
    void secItemRequestSync(IPC::Connection&, const SecItemRequestData&, CompletionHandler<void(std::optional<SecItemResponseData>&&)>&&);

    Ref<WorkQueue> m_queue;
};

} // namespace WebKit

#endif // ENABLE(SEC_ITEM_SHIM)
