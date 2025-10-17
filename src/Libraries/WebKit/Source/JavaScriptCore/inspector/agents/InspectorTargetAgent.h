/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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

#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "InspectorFrontendChannel.h"
#include "InspectorFrontendDispatchers.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace Inspector {

class InspectorTarget;

class JS_EXPORT_PRIVATE InspectorTargetAgent final : public InspectorAgentBase, public TargetBackendDispatcherHandler, public CanMakeCheckedPtr<InspectorTargetAgent> {
    WTF_MAKE_NONCOPYABLE(InspectorTargetAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorTargetAgent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorTargetAgent);
public:
    InspectorTargetAgent(FrontendRouter&, BackendDispatcher&);
    ~InspectorTargetAgent() final;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // TargetBackendDispatcherHandler
    Protocol::ErrorStringOr<void> setPauseOnStart(bool) final;
    Protocol::ErrorStringOr<void> resume(const String& targetId) final;
    Protocol::ErrorStringOr<void> sendMessageToTarget(const String& targetId, const String& message) final;

    // Target lifecycle.
    void targetCreated(InspectorTarget&);
    void targetDestroyed(InspectorTarget&);
    void didCommitProvisionalTarget(const String& oldTargetID, const String& committedTargetID);

    // Target messages.
    void sendMessageFromTargetToFrontend(const String& targetId, const String& message);

private:
    // FrontendChannel
    FrontendChannel::ConnectionType connectionType() const;
    void connectToTargets();
    void disconnectFromTargets();

    Inspector::FrontendRouter& m_router;
    std::unique_ptr<TargetFrontendDispatcher> m_frontendDispatcher;
    Ref<TargetBackendDispatcher> m_backendDispatcher;
    UncheckedKeyHashMap<String, InspectorTarget*> m_targets;
    bool m_isConnected { false };
    bool m_shouldPauseOnStart { false };
};

} // namespace Inspector
