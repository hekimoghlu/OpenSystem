/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#include "MessageReceiver.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
enum class PermissionQuerySource : uint8_t;
enum class PermissionState : uint8_t;
struct ClientOrigin;
struct PermissionDescriptor;
class SecurityOriginData;
}

namespace WebKit {

class WebPageProxy;
class WebProcessProxy;

class WebPermissionControllerProxy final : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebPermissionControllerProxy);
public:
    explicit WebPermissionControllerProxy(WebProcessProxy&);
    ~WebPermissionControllerProxy();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    void ref() const final;
    void deref() const final;

private:
    RefPtr<WebPageProxy> mostReasonableWebPageProxy(const WebCore::SecurityOriginData&, WebCore::PermissionQuerySource) const;

    // IPC Message handlers.
    void query(const WebCore::ClientOrigin&, const WebCore::PermissionDescriptor&, std::optional<WebPageProxyIdentifier>, WebCore::PermissionQuerySource, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&&);

    Ref<WebProcessProxy> protectedProcess() const;

    CheckedRef<WebProcessProxy> m_process;
};

} // namespace WebKit
