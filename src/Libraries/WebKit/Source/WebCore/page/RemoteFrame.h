/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include "Frame.h"
#include "LayerHostingContextIdentifier.h"
#include <wtf/RefPtr.h>
#include <wtf/TypeCasts.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class IntPoint;
class RemoteDOMWindow;
class RemoteFrameClient;
class RemoteFrameView;
class WeakPtrImplWithEventTargetData;

enum class AdvancedPrivacyProtections : uint16_t;
enum class RenderAsTextFlag : uint16_t;

class RemoteFrame final : public Frame {
public:
    using ClientCreator = CompletionHandler<UniqueRef<RemoteFrameClient>(RemoteFrame&)>;
    WEBCORE_EXPORT static Ref<RemoteFrame> createMainFrame(Page&, ClientCreator&&, FrameIdentifier, Frame* opener);
    WEBCORE_EXPORT static Ref<RemoteFrame> createSubframe(Page&, ClientCreator&&, FrameIdentifier, Frame& parent, Frame* opener);
    WEBCORE_EXPORT static Ref<RemoteFrame> createSubframeWithContentsInAnotherProcess(Page&, ClientCreator&&, FrameIdentifier, HTMLFrameOwnerElement&, std::optional<LayerHostingContextIdentifier>);
    ~RemoteFrame();

    RemoteDOMWindow& window() const;

    const RemoteFrameClient& client() const { return m_client.get(); }
    RemoteFrameClient& client() { return m_client.get(); }

    RemoteFrameView* view() const { return m_view.get(); }
    WEBCORE_EXPORT void setView(RefPtr<RemoteFrameView>&&);

    Markable<LayerHostingContextIdentifier> layerHostingContextIdentifier() const { return m_layerHostingContextIdentifier; }

    String renderTreeAsText(size_t baseIndent, OptionSet<RenderAsTextFlag>);
    void bindRemoteAccessibilityFrames(int processIdentifier, Vector<uint8_t>&&, CompletionHandler<void(Vector<uint8_t>, int)>&&);
    void updateRemoteFrameAccessibilityOffset(IntPoint);
    void unbindRemoteAccessibilityFrames(int);

    void setCustomUserAgent(const String& customUserAgent) { m_customUserAgent = customUserAgent; }
    String customUserAgent() const final;
    void setCustomUserAgentAsSiteSpecificQuirks(const String& customUserAgentAsSiteSpecificQuirks) { m_customUserAgentAsSiteSpecificQuirks = customUserAgentAsSiteSpecificQuirks; }
    String customUserAgentAsSiteSpecificQuirks() const final;

    void setCustomNavigatorPlatform(const String& customNavigatorPlatform) { m_customNavigatorPlatform = customNavigatorPlatform; }
    String customNavigatorPlatform() const final;

    void setAdvancedPrivacyProtections(OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections) { m_advancedPrivacyProtections = advancedPrivacyProtections; }
    OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections() const final;

    void updateScrollingMode() final;

private:
    WEBCORE_EXPORT explicit RemoteFrame(Page&, ClientCreator&&, FrameIdentifier, HTMLFrameOwnerElement*, Frame* parent, Markable<LayerHostingContextIdentifier>, Frame* opener);

    void frameDetached() final;
    bool preventsParentFromBeingComplete() const final;
    void changeLocation(FrameLoadRequest&&) final;
    void didFinishLoadInAnotherProcess() final;
    bool isRootFrame() const final { return false; }
    void documentURLForConsoleLog(CompletionHandler<void(const URL&)>&&) final;

    FrameView* virtualView() const final;
    void disconnectView() final;
    DOMWindow* virtualWindow() const final;
    FrameLoaderClient& loaderClient() final;
    void reinitializeDocumentSecurityContext() final { }

    Ref<RemoteDOMWindow> m_window;
    RefPtr<RemoteFrameView> m_view;
    UniqueRef<RemoteFrameClient> m_client;
    Markable<LayerHostingContextIdentifier> m_layerHostingContextIdentifier;
    String m_customUserAgent;
    String m_customUserAgentAsSiteSpecificQuirks;
    String m_customNavigatorPlatform;
    OptionSet<AdvancedPrivacyProtections> m_advancedPrivacyProtections;
    bool m_preventsParentFromBeingComplete { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RemoteFrame)
static bool isType(const WebCore::Frame& frame) { return frame.frameType() == WebCore::Frame::FrameType::Remote; }
SPECIALIZE_TYPE_TRAITS_END()
