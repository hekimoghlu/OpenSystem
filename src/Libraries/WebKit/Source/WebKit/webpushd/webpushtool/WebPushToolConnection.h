/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#include "MessageSenderInlines.h"
#include "PushMessageForTesting.h"
#include <WebCore/PushPermissionState.h>
#include <memory>
#include <wtf/CompletionHandler.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/spi/darwin/XPCSPI.h>

using WebKit::WebPushD::PushMessageForTesting;

namespace WebPushTool {

enum class PreferTestService : bool {
    No,
    Yes,
};

enum class WaitForServiceToExist : bool {
    No,
    Yes,
};

class Connection final : public RefCountedAndCanMakeWeakPtr<Connection>, public IPC::MessageSender {
    WTF_MAKE_TZONE_ALLOCATED(Connection);
public:
    static Ref<Connection> create(PreferTestService, String bundleIdentifier, String pushPartition);
    Connection(PreferTestService, String bundleIdentifier, String pushPartition);
    ~Connection() final { }

    void connectToService(WaitForServiceToExist);

    String bundleIdentifier() const { return m_bundleIdentifier; }
    String pushPartition() const { return m_pushPartition; }

    void sendPushMessage(PushMessageForTesting&&, CompletionHandler<void(String)>&&);
    void getPushPermissionState(const String& scope, CompletionHandler<void(WebCore::PushPermissionState)>&&);
    void requestPushPermission(const String& scope, CompletionHandler<void(bool)>&&);

private:
    void sendAuditToken();

    bool performSendWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&&) const final;
    bool performSendWithAsyncReplyWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&&, CompletionHandler<void(IPC::Decoder*)>&&) const final;
    IPC::Connection* messageSenderConnection() const final { return nullptr; }
    uint64_t messageSenderDestinationID() const final { return 0; }

    String m_bundleIdentifier;
    String m_pushPartition;

    RetainPtr<xpc_connection_t> m_connection;
    ASCIILiteral m_serviceName;
};

} // namespace WebPushTool
