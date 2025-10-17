/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

#include "HTMLFrameElementBase.h"
#include "PermissionsPolicy.h"

namespace WebCore {

class DOMTokenList;
class LazyLoadFrameObserver;
class RenderIFrame;
class TrustedHTML;

class HTMLIFrameElement final : public HTMLFrameElementBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLIFrameElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLIFrameElement);
public:
    static Ref<HTMLIFrameElement> create(const QualifiedName&, Document&);
    ~HTMLIFrameElement();

    DOMTokenList& sandbox();

    void setReferrerPolicyForBindings(const AtomString&);
    String referrerPolicyForBindings() const;
    ReferrerPolicy referrerPolicy() const final;

    const AtomString& loading() const;
    void setLoading(const AtomString&);

    String srcdoc() const;
    ExceptionOr<void> setSrcdoc(std::variant<RefPtr<TrustedHTML>, String>&&);

    LazyLoadFrameObserver& lazyLoadFrameObserver();

    void loadDeferredFrame();

#if ENABLE(FULLSCREEN_API)
    bool hasIFrameFullscreenFlag() const { return m_IFrameFullscreenFlag; }
    void setIFrameFullscreenFlag(bool value) { m_IFrameFullscreenFlag = value; }
#endif

#if ENABLE(CONTENT_EXTENSIONS)
    const URL& initiatorSourceURL() const { return m_initiatorSourceURL; }
    void setInitiatorSourceURL(URL&& url) { m_initiatorSourceURL = WTFMove(url); }
#endif

private:
    HTMLIFrameElement(const QualifiedName&, Document&);

    int defaultTabIndex() const final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    bool isInteractiveContent() const final { return true; }

    bool rendererIsNeeded(const RenderStyle&) final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool isReplaced(const RenderStyle&) const final { return true; }

    bool shouldLoadFrameLazily() final;
    bool isLazyLoadObserverActive() const final;

    std::unique_ptr<DOMTokenList> m_sandbox;
#if ENABLE(FULLSCREEN_API)
    bool m_IFrameFullscreenFlag { false };
#endif
    std::unique_ptr<LazyLoadFrameObserver> m_lazyLoadFrameObserver;
#if ENABLE(CONTENT_EXTENSIONS)
    URL m_initiatorSourceURL;
#endif
};

} // namespace WebCore
