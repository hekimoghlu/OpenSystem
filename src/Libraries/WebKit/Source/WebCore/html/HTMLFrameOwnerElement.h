/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#include "HTMLElement.h"
#include "ReferrerPolicy.h"
#include "SecurityContext.h"
#include <wtf/HashCountedSet.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class RenderWidget;

class HTMLFrameOwnerElement : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFrameOwnerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLFrameOwnerElement);
public:
    virtual ~HTMLFrameOwnerElement();

    Frame* contentFrame() const { return m_contentFrame.get(); }
    RefPtr<Frame> protectedContentFrame() const;
    WEBCORE_EXPORT WindowProxy* contentWindow() const;
    WEBCORE_EXPORT Document* contentDocument() const;
    RefPtr<Document> protectedContentDocument() const { return contentDocument(); }

    WEBCORE_EXPORT void setContentFrame(Frame&);
    void clearContentFrame();

    void disconnectContentFrame();

    // Most subclasses use RenderWidget (either RenderEmbeddedObject or RenderIFrame)
    // except for HTMLObjectElement and HTMLEmbedElement which may return any
    // RenderElement when using fallback content.
    RenderWidget* renderWidget() const;

    Document* getSVGDocument() const;

    virtual ScrollbarMode scrollingMode() const { return ScrollbarMode::Auto; }

    SandboxFlags sandboxFlags() const { return m_sandboxFlags; }

    WEBCORE_EXPORT void scheduleInvalidateStyleAndLayerComposition();

    virtual bool canLoadScriptURL(const URL&) const = 0;

    virtual ReferrerPolicy referrerPolicy() const { return ReferrerPolicy::EmptyString; }

    virtual bool shouldLoadFrameLazily() { return false; }
    virtual bool isLazyLoadObserverActive() const { return false; }

protected:
    HTMLFrameOwnerElement(const QualifiedName& tagName, Document&, OptionSet<TypeFlag> = { });
    void setSandboxFlags(SandboxFlags);
    bool isProhibitedSelfReference(const URL&) const;
    bool isKeyboardFocusable(KeyboardEvent*) const override;

private:
    bool isHTMLFrameOwnerElement() const final { return true; }

    WeakPtr<Frame> m_contentFrame;
    SandboxFlags m_sandboxFlags;
};

class SubframeLoadingDisabler {
public:
    explicit SubframeLoadingDisabler(ContainerNode* root)
        : m_root(root)
    {
        if (m_root)
            disabledSubtreeRoots().add(m_root.get());
    }

    ~SubframeLoadingDisabler()
    {
        if (m_root)
            disabledSubtreeRoots().remove(m_root.get());
    }

    static bool canLoadFrame(HTMLFrameOwnerElement&);

private:
    static HashCountedSet<ContainerNode*>& disabledSubtreeRoots()
    {
        static NeverDestroyed<HashCountedSet<ContainerNode*>> nodes;
        return nodes;
    }

    WeakPtr<ContainerNode, WeakPtrImplWithEventTargetData> m_root;
};

inline HTMLFrameOwnerElement* Frame::ownerElement() const
{
    return m_ownerElement.get();
}

inline RefPtr<HTMLFrameOwnerElement> Frame::protectedOwnerElement() const
{
    return m_ownerElement.get();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLFrameOwnerElement)
    static bool isType(const WebCore::Node& node) { return node.isHTMLFrameOwnerElement(); }
SPECIALIZE_TYPE_TRAITS_END()
