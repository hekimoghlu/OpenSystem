/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#include "FrameIdentifier.h"
#include "FrameTree.h"
#include "OwnerPermissionsPolicyData.h"
#include "PageIdentifier.h"
#include <wtf/CheckedRef.h>
#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMWindow;
class FrameView;
class FrameLoaderClient;
class FrameLoadRequest;
class HTMLFrameOwnerElement;
class NavigationScheduler;
class Page;
class RenderWidget;
class Settings;
class WeakPtrImplWithEventTargetData;
class WindowProxy;

enum class AdvancedPrivacyProtections : uint16_t;
enum class SandboxFlag : uint16_t;
enum class ScrollbarMode : uint8_t;

using SandboxFlags = OptionSet<SandboxFlag>;

class Frame : public ThreadSafeRefCounted<Frame, WTF::DestructionThread::Main>, public CanMakeWeakPtr<Frame> {
public:
    virtual ~Frame();

    enum class NotifyUIProcess : bool { No, Yes };
    enum class FrameType : bool { Local, Remote };
    FrameType frameType() const { return m_frameType; }

    WindowProxy& windowProxy() { return m_windowProxy; }
    const WindowProxy& windowProxy() const { return m_windowProxy; }
    Ref<WindowProxy> protectedWindowProxy() const;

    DOMWindow* window() const { return virtualWindow(); }
    FrameTree& tree() const { return m_treeNode; }
    FrameIdentifier frameID() const { return m_frameID; }
    inline Page* page() const; // Defined in Page.h.
    inline RefPtr<Page> protectedPage() const; // Defined in Page.h.
    WEBCORE_EXPORT std::optional<PageIdentifier> pageID() const;
    Settings& settings() const { return m_settings.get(); }
    Frame& mainFrame() { return *m_mainFrame; }
    const Frame& mainFrame() const { return *m_mainFrame; }
    bool isMainFrame() const { return this == m_mainFrame.get(); }
    WEBCORE_EXPORT void disownOpener();
    WEBCORE_EXPORT void updateOpener(Frame&, NotifyUIProcess = NotifyUIProcess::Yes);
    WEBCORE_EXPORT void setOpenerForWebKitLegacy(Frame*);
    const Frame* opener() const { return m_opener.get(); }
    Frame* opener() { return m_opener.get(); }
    bool hasOpenedFrames() const;
    WEBCORE_EXPORT void detachFromAllOpenedFrames();
    virtual bool isRootFrame() const = 0;
#if ASSERT_ENABLED
    WEBCORE_EXPORT static bool isRootFrameIdentifier(FrameIdentifier);
#endif

    WEBCORE_EXPORT void detachFromPage();

    WEBCORE_EXPORT void setOwnerElement(HTMLFrameOwnerElement*);
    inline HTMLFrameOwnerElement* ownerElement() const; // Defined in HTMLFrameOwnerElement.h.
    inline RefPtr<HTMLFrameOwnerElement> protectedOwnerElement() const; // Defined in HTMLFrameOwnerElement.h.

    WEBCORE_EXPORT void disconnectOwnerElement();
    NavigationScheduler& navigationScheduler() const { return m_navigationScheduler.get(); }
    Ref<NavigationScheduler> protectedNavigationScheduler() const;
    WEBCORE_EXPORT void takeWindowProxyAndOpenerFrom(Frame&);

    virtual void frameDetached() = 0;
    virtual bool preventsParentFromBeingComplete() const = 0;
    virtual void changeLocation(FrameLoadRequest&&) = 0;
    virtual void didFinishLoadInAnotherProcess() = 0;

    virtual FrameView* virtualView() const = 0;
    RefPtr<FrameView> protectedVirtualView() const;
    virtual void disconnectView() = 0;
    virtual FrameLoaderClient& loaderClient() = 0;
    virtual void documentURLForConsoleLog(CompletionHandler<void(const URL&)>&&) = 0;

    virtual String customUserAgent() const = 0;
    virtual String customUserAgentAsSiteSpecificQuirks() const = 0;
    virtual String customNavigatorPlatform() const = 0;
    virtual OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections() const = 0;

    virtual void updateSandboxFlags(SandboxFlags, NotifyUIProcess);

    WEBCORE_EXPORT RenderWidget* ownerRenderer() const; // Renderer for the element that contains this frame.

    WEBCORE_EXPORT void setOwnerPermissionsPolicy(OwnerPermissionsPolicyData&&);
    WEBCORE_EXPORT std::optional<OwnerPermissionsPolicyData> ownerPermissionsPolicy() const;

    virtual void updateScrollingMode() = 0;

protected:
    Frame(Page&, FrameIdentifier, FrameType, HTMLFrameOwnerElement*, Frame* parent, Frame* opener);
    void resetWindowProxy();

    virtual void frameWasDisconnectedFromOwner() const { }

private:
    virtual DOMWindow* virtualWindow() const = 0;
    virtual void reinitializeDocumentSecurityContext() = 0;

    WeakPtr<Page> m_page;
    const FrameIdentifier m_frameID;
    mutable FrameTree m_treeNode;
    Ref<WindowProxy> m_windowProxy;
    WeakPtr<HTMLFrameOwnerElement, WeakPtrImplWithEventTargetData> m_ownerElement;
    const WeakPtr<Frame> m_mainFrame;
    const Ref<Settings> m_settings;
    FrameType m_frameType;
    mutable UniqueRef<NavigationScheduler> m_navigationScheduler;
    WeakPtr<Frame> m_opener;
    WeakHashSet<Frame> m_openedFrames;
    std::optional<OwnerPermissionsPolicyData> m_ownerPermisssionsPolicyOverride;
};

} // namespace WebCore
