/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include "WebPageInspectorTarget.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class InspectorTarget;
}

namespace WebKit {

class WebPage;

class WebPageInspectorTargetController {
    WTF_MAKE_TZONE_ALLOCATED(WebPageInspectorTargetController);
public:
    WebPageInspectorTargetController(WebPage&);
    ~WebPageInspectorTargetController();

    void addTarget(Inspector::InspectorTarget&);
    void removeTarget(Inspector::InspectorTarget&);

    void connectInspector(const String& targetId, Inspector::FrontendChannel::ConnectionType);
    void disconnectInspector(const String& targetId);
    void sendMessageToTargetBackend(const String& targetId, const String& message);

private:
    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;
    WebPageInspectorTarget m_pageTarget;
    HashMap<String, WeakPtr<Inspector::InspectorTarget>> m_targets;
};

} // namespace WebKit
