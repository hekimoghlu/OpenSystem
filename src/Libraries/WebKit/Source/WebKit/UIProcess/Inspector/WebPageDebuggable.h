/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#if ENABLE(REMOTE_INSPECTOR)

#include <JavaScriptCore/RemoteInspectionTarget.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class WebPageProxy;

class WebPageDebuggable final : public Inspector::RemoteInspectionTarget {
    WTF_MAKE_TZONE_ALLOCATED(WebPageDebuggable);
    WTF_MAKE_NONCOPYABLE(WebPageDebuggable);
public:
    static Ref<WebPageDebuggable> create(WebPageProxy&);
    ~WebPageDebuggable();

    Inspector::RemoteControllableTarget::Type type() const final { return Inspector::RemoteControllableTarget::Type::WebPage; }

    String name() const final;
    String url() const final;
    bool hasLocalDebugger() const final;

    void connect(Inspector::FrontendChannel&, bool isAutomaticConnection = false, bool immediatelyPause = false) final;
    void disconnect(Inspector::FrontendChannel&) final;
    void dispatchMessageFromRemote(String&& message) final;
    void setIndicating(bool) final;

    const String& nameOverride() const final { return m_nameOverride; }
    void setNameOverride(const String&);

    void detachFromPage();

private:
    explicit WebPageDebuggable(WebPageProxy&);

    WeakPtr<WebPageProxy> m_page;
    String m_nameOverride;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_CONTROLLABLE_TARGET(WebKit::WebPageDebuggable, WebPage);

#endif // ENABLE(REMOTE_INSPECTOR)
