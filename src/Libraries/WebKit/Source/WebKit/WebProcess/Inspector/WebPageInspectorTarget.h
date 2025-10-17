/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include <WebCore/PageIdentifier.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;
class WebPageInspectorTargetFrontendChannel;

class WebPageInspectorTarget final : public Inspector::InspectorTarget {
    WTF_MAKE_TZONE_ALLOCATED(WebPageInspectorTarget);
    WTF_MAKE_NONCOPYABLE(WebPageInspectorTarget);
public:
    explicit WebPageInspectorTarget(WebPage&);
    virtual ~WebPageInspectorTarget();

    Inspector::InspectorTargetType type() const final { return Inspector::InspectorTargetType::Page; }

    String identifier() const final;

    void connect(Inspector::FrontendChannel::ConnectionType) override;
    void disconnect() override;
    void sendMessageToTargetBackend(const String&) override;

    static String toTargetID(WebCore::PageIdentifier);

private:
    WeakRef<WebPage> m_page;
    std::unique_ptr<WebPageInspectorTargetFrontendChannel> m_channel;
};

} // namespace WebKit
