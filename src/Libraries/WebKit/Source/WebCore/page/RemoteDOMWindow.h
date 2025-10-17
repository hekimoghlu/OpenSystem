/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include "DOMWindow.h"
#include "RemoteFrame.h"
#include "WindowPostMessageOptions.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
class JSObject;
class JSValue;
}

namespace WebCore {

class LocalDOMWindow;
class Document;
class Location;

class RemoteDOMWindow final : public DOMWindow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(RemoteDOMWindow, WEBCORE_EXPORT);
public:
    static Ref<RemoteDOMWindow> create(RemoteFrame& frame, GlobalWindowIdentifier&& identifier)
    {
        return adoptRef(*new RemoteDOMWindow(frame, WTFMove(identifier)));
    }

    ~RemoteDOMWindow() final;

    RemoteFrame* frame() const final { return m_frame.get(); }
    ScriptExecutionContext* scriptExecutionContext() const final { return nullptr; }

    // DOM API exposed cross-origin.
    WindowProxy* self() const;
    void focus(LocalDOMWindow& incumbentWindow);
    void blur();
    unsigned length() const;
    void frameDetached();
    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, LocalDOMWindow& incumbentWindow, JSC::JSValue message, WindowPostMessageOptions&&);

private:
    WEBCORE_EXPORT RemoteDOMWindow(RemoteFrame&, GlobalWindowIdentifier&&);

    void closePage() final;
    void setLocation(LocalDOMWindow& activeWindow, const URL& completedURL, NavigationHistoryBehavior, SetLocationLocking) final;

    WeakPtr<RemoteFrame> m_frame;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RemoteDOMWindow)
    static bool isType(const WebCore::DOMWindow& window) { return window.isRemoteDOMWindow(); }
SPECIALIZE_TYPE_TRAITS_END()
