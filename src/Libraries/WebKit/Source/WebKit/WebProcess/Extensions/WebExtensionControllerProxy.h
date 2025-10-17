/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "MessageReceiver.h"
#include "WebExtensionContextProxy.h"
#include "WebExtensionControllerParameters.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>

namespace WebCore {
class DOMWrapperWorld;
}

namespace WebKit {

class WebFrame;
class WebPage;

class WebExtensionControllerProxy final : public RefCounted<WebExtensionControllerProxy>, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionControllerProxy);
    WTF_MAKE_NONCOPYABLE(WebExtensionControllerProxy);

public:
    static RefPtr<WebExtensionControllerProxy> get(WebExtensionControllerIdentifier);
    static Ref<WebExtensionControllerProxy> getOrCreate(const WebExtensionControllerParameters&, WebPage* = nullptr);

    ~WebExtensionControllerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using WebExtensionContextProxySet = HashSet<Ref<WebExtensionContextProxy>>;
    using WebExtensionContextProxyBaseURLMap = HashMap<String, Ref<WebExtensionContextProxy>>;

    WebExtensionControllerIdentifier identifier() { return m_identifier; }

    bool operator==(const WebExtensionControllerProxy& other) const { return (this == &other); }

    bool inTestingMode() { return m_testingMode; }

    void globalObjectIsAvailableForFrame(WebPage&, WebFrame&, WebCore::DOMWrapperWorld&);
    void serviceWorkerGlobalObjectIsAvailableForFrame(WebPage&, WebFrame&, WebCore::DOMWrapperWorld&);
    void addBindingsToWebPageFrameIfNecessary(WebFrame&, WebCore::DOMWrapperWorld&);

    void didStartProvisionalLoadForFrame(WebPage&, WebFrame&, const URL&);
    void didCommitLoadForFrame(WebPage&, WebFrame&, const URL&);
    void didFinishLoadForFrame(WebPage&, WebFrame&, const URL&);
    // FIXME: Include the error here.
    void didFailLoadForFrame(WebPage&, WebFrame&, const URL&);

    RefPtr<WebExtensionContextProxy> extensionContext(const String& uniqueIdentifier) const;
    RefPtr<WebExtensionContextProxy> extensionContext(const URL&) const;
    RefPtr<WebExtensionContextProxy> extensionContext(WebFrame&, WebCore::DOMWrapperWorld&) const;

    bool hasLoadedContexts() const { return !m_extensionContexts.isEmpty(); }
    const WebExtensionContextProxySet& extensionContexts() const { return m_extensionContexts; }

private:
    explicit WebExtensionControllerProxy(const WebExtensionControllerParameters&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void load(const WebExtensionContextParameters&);
    void unload(WebExtensionContextIdentifier);

    WebExtensionControllerIdentifier m_identifier;
    bool m_testingMode { false };

    WebExtensionContextProxySet m_extensionContexts;
    WebExtensionContextProxyBaseURLMap m_extensionContextBaseURLMap;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
