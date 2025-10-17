/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include <WebCore/ClientOrigin.h>
#include <WebCore/PermissionController.h>
#include <WebCore/PermissionDescriptor.h>
#include <wtf/Deque.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
enum class PermissionQuerySource : uint8_t;
enum class PermissionState : uint8_t;
class Page;
class SecurityOriginData;
}

namespace WebKit {

class WebProcess;

class WebPermissionController final : public WebCore::PermissionController, public IPC::MessageReceiver {
public:
    static Ref<WebPermissionController> create(WebProcess&);
    ~WebPermissionController();

    void ref() const final { WebCore::PermissionController::ref(); }
    void deref() const final { WebCore::PermissionController::deref(); }

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    explicit WebPermissionController(WebProcess&);

    // WebCore::PermissionController
    void query(WebCore::ClientOrigin&&, WebCore::PermissionDescriptor, const WeakPtr<WebCore::Page>&, WebCore::PermissionQuerySource, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&&) final;
    void addObserver(WebCore::PermissionObserver&) final;
    void removeObserver(WebCore::PermissionObserver&) final;
    void permissionChanged(WebCore::PermissionName, const WebCore::SecurityOriginData&) final;

    WeakHashSet<WebCore::PermissionObserver> m_observers;
};

} // namespace WebCore
