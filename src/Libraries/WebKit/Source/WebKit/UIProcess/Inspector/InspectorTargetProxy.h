/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include <JavaScriptCore/InspectorTarget.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class ProvisionalPageProxy;
class WebPageProxy;

// NOTE: This UIProcess side InspectorTarget doesn't care about the frontend channel, since
// any target -> frontend messages will be routed to the WebPageProxy with a targetId.

class InspectorTargetProxy final : public Inspector::InspectorTarget {
    WTF_MAKE_TZONE_ALLOCATED(InspectorTargetProxy);
    WTF_MAKE_NONCOPYABLE(InspectorTargetProxy);
public:
    static std::unique_ptr<InspectorTargetProxy> create(WebPageProxy&, const String& targetId, Inspector::InspectorTargetType);
    static std::unique_ptr<InspectorTargetProxy> create(ProvisionalPageProxy&, const String& targetId, Inspector::InspectorTargetType);
    InspectorTargetProxy(WebPageProxy&, const String& targetId, Inspector::InspectorTargetType);
    ~InspectorTargetProxy() = default;

    Inspector::InspectorTargetType type() const final { return m_type; }
    String identifier() const final { return m_identifier; }

    void didCommitProvisionalTarget();
    bool isProvisional() const override;

    void connect(Inspector::FrontendChannel::ConnectionType) override;
    void disconnect() override;
    void sendMessageToTargetBackend(const String&) override;

private:
    WeakRef<WebPageProxy> m_page;
    String m_identifier;
    Inspector::InspectorTargetType m_type;
    WeakPtr<ProvisionalPageProxy> m_provisionalPage;
};

} // namespace WebKit
