/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#include "Connection.h"
#include "CoordinateSystem.h"
#include <JavaScriptCore/JSBase.h>
#include <JavaScriptCore/PrivateName.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/IntRect.h>
#include <WebCore/PageIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

#if ENABLE(WEBDRIVER_BIDI)
#include <JavaScriptCore/ConsoleMessage.h>
#include <WebCore/AutomationInstrumentation.h>
#endif

namespace WebCore {
struct Cookie;
class AccessibilityObject;
class Element;
class ShareableBitmapHandle;
}

namespace WebKit {

class WebFrame;
class WebPage;
class WebAutomationDOMWindowObserver;

class WebAutomationSessionProxy : public IPC::MessageReceiver
#if ENABLE(WEBDRIVER_BIDI)
    , public WebCore::AutomationInstrumentationClient
#endif
    , public ThreadSafeRefCounted<WebAutomationSessionProxy> {
    WTF_MAKE_TZONE_ALLOCATED(WebAutomationSessionProxy);
public:
    static Ref<WebAutomationSessionProxy> create(const String& sessionIdentifier);
    ~WebAutomationSessionProxy();

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    String sessionIdentifier() const { return m_sessionIdentifier; }

    void didClearWindowObjectForFrame(WebFrame&);
    void willDestroyGlobalObjectForFrame(WebCore::FrameIdentifier);

    struct JSCallbackIdentifierType { };
    using JSCallbackIdentifier = ObjectIdentifier<JSCallbackIdentifierType>;

    void didEvaluateJavaScriptFunction(WebCore::FrameIdentifier, JSCallbackIdentifier, const String& result, const String& errorType);

private:
    explicit WebAutomationSessionProxy(const String& sessionIdentifier);
    JSObjectRef scriptObject(JSGlobalContextRef);
    void setScriptObject(JSGlobalContextRef, JSObjectRef);
    JSObjectRef scriptObjectForFrame(WebFrame&);
    WebCore::Element* elementForNodeHandle(WebFrame&, const String&);
    WebCore::AccessibilityObject* getAccessibilityObjectForNode(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, String& error);

    void ensureObserverForFrame(WebFrame&);

    // Implemented in generated WebAutomationSessionProxyMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Called by WebAutomationSessionProxy messages
    void evaluateJavaScriptFunction(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, const String& function, Vector<String> arguments, bool expectsImplicitCallbackArgument, bool forceUserGesture, std::optional<double> callbackTimeout, CompletionHandler<void(String&&, String&&)>&&);
    void resolveChildFrameWithOrdinal(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, uint32_t ordinal, CompletionHandler<void(std::optional<String>, std::optional<WebCore::FrameIdentifier>)>&&);
    void resolveChildFrameWithNodeHandle(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, const String& nodeHandle, CompletionHandler<void(std::optional<String>, std::optional<WebCore::FrameIdentifier>)>&&);
    void resolveChildFrameWithName(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, const String& name, CompletionHandler<void(std::optional<String>, std::optional<WebCore::FrameIdentifier>)>&&);
    void resolveParentFrame(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, CompletionHandler<void(std::optional<String>, std::optional<WebCore::FrameIdentifier>)>&&);
    void computeElementLayout(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, bool scrollIntoViewIfNeeded, CoordinateSystem, CompletionHandler<void(std::optional<String>, WebCore::FloatRect, std::optional<WebCore::IntPoint>, bool)>&&);
    void getComputedRole(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, CompletionHandler<void(std::optional<String>, std::optional<String>)>&&);
    void getComputedLabel(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, CompletionHandler<void(std::optional<String>, std::optional<String>)>&&);
    void selectOptionElement(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, CompletionHandler<void(std::optional<String>)>&&);
    void setFilesForInputFileUpload(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, Vector<String>&& filenames, CompletionHandler<void(std::optional<String>)>&&);
    void takeScreenshot(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, bool scrollIntoViewIfNeeded, bool clipToViewport, CompletionHandler<void(std::optional<WebCore::ShareableBitmapHandle>&&, String&&)>&&);
    void snapshotRectForScreenshot(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String nodeHandle, bool scrollIntoViewIfNeeded, bool clipToViewport, CompletionHandler<void(std::optional<String>, WebCore::IntRect&&)>&&);
    void getCookiesForFrame(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, CompletionHandler<void(std::optional<String>, Vector<WebCore::Cookie>)>&&);
    void deleteCookie(WebCore::PageIdentifier, std::optional<WebCore::FrameIdentifier>, String cookieName, CompletionHandler<void(std::optional<String>)>&&);

#if ENABLE(WEBDRIVER_BIDI)
    void addMessageToConsole(const JSC::MessageSource&, const JSC::MessageLevel&, const String&, const JSC::MessageType&, const WallTime&) override;
#endif

    String m_sessionIdentifier;
    JSC::PrivateName m_scriptObjectIdentifier;

    HashMap<WebCore::FrameIdentifier, HashMap<JSCallbackIdentifier, CompletionHandler<void(String&&, String&&)>>> m_webFramePendingEvaluateJavaScriptCallbacksMap;
    HashMap<WebCore::FrameIdentifier, RefPtr<WebAutomationDOMWindowObserver>> m_frameObservers;
};

} // namespace WebKit
