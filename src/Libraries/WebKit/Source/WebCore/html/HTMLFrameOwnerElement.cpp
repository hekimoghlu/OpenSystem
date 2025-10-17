/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "config.h"
#include "HTMLFrameOwnerElement.h"

#include "FrameLoader.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "RemoteFrame.h"
#include "RemoteFrameClient.h"
#include "RenderWidget.h"
#include "SVGDocument.h"
#include "SVGElementTypeHelpers.h"
#include "ScriptController.h"
#include "ShadowRoot.h"
#include "StyleTreeResolver.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFrameOwnerElement);

HTMLFrameOwnerElement::HTMLFrameOwnerElement(const QualifiedName& tagName, Document& document, OptionSet<TypeFlag> constructionType)
    : HTMLElement(tagName, document, constructionType)
{
}

RenderWidget* HTMLFrameOwnerElement::renderWidget() const
{
    // HTMLObjectElement and HTMLEmbedElement may return arbitrary renderers
    // when using fallback content.
    return dynamicDowncast<RenderWidget>(renderer());
}

void HTMLFrameOwnerElement::setContentFrame(Frame& frame)
{
    // Make sure we will not end up with two frames referencing the same owner element.
    ASSERT(!m_contentFrame || m_contentFrame->ownerElement() != this);
    // Disconnected frames should not be allowed to load.
    ASSERT(isConnected());
    m_contentFrame = frame;

    for (RefPtr<ContainerNode> node = this; node; node = node->parentOrShadowHostNode())
        node->incrementConnectedSubframeCount();
}

void HTMLFrameOwnerElement::clearContentFrame()
{
    if (!m_contentFrame)
        return;

    m_contentFrame = nullptr;

    for (RefPtr<ContainerNode> node = this; node; node = node->parentOrShadowHostNode())
        node->decrementConnectedSubframeCount();
}

void HTMLFrameOwnerElement::disconnectContentFrame()
{
    if (RefPtr frame = m_contentFrame.get()) {
        frame->frameDetached();
        if (frame == m_contentFrame.get())
            frame->disconnectOwnerElement();
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(frame != m_contentFrame.get());
    }
}

HTMLFrameOwnerElement::~HTMLFrameOwnerElement()
{
    if (RefPtr contentFrame = m_contentFrame.get())
        contentFrame->disconnectOwnerElement();
}

RefPtr<Frame> HTMLFrameOwnerElement::protectedContentFrame() const
{
    return m_contentFrame.get();
}

Document* HTMLFrameOwnerElement::contentDocument() const
{
    if (auto* localFrame = dynamicDowncast<LocalFrame>(m_contentFrame.get()))
        return localFrame->document();
    return nullptr;
}

WindowProxy* HTMLFrameOwnerElement::contentWindow() const
{
    return m_contentFrame ? &m_contentFrame->windowProxy() : nullptr;
}

void HTMLFrameOwnerElement::setSandboxFlags(SandboxFlags flags)
{
    m_sandboxFlags = flags;
    if (m_contentFrame)
        m_contentFrame->updateSandboxFlags(flags, Frame::NotifyUIProcess::Yes);
}

bool HTMLFrameOwnerElement::isKeyboardFocusable(KeyboardEvent* event) const
{
    return m_contentFrame && HTMLElement::isKeyboardFocusable(event);
}

Document* HTMLFrameOwnerElement::getSVGDocument() const
{
    auto* document = contentDocument();
    if (is<SVGDocument>(document))
        return document;
    return nullptr;
}

void HTMLFrameOwnerElement::scheduleInvalidateStyleAndLayerComposition()
{
    if (Style::postResolutionCallbacksAreSuspended()) {
        RefPtr<HTMLFrameOwnerElement> element = this;
        Style::deprecatedQueuePostResolutionCallback([element] {
            element->invalidateStyleAndLayerComposition();
        });
    } else
        invalidateStyleAndLayerComposition();
}

bool HTMLFrameOwnerElement::isProhibitedSelfReference(const URL& completeURL) const
{
    // We allow one level of self-reference because some websites depend on that, but we don't allow more than one.
    bool foundOneSelfReference = false;
    for (Frame* frame = document().frame(); frame; frame = frame->tree().parent()) {
        auto* localFrame = dynamicDowncast<LocalFrame>(frame);
        if (!localFrame)
            continue;
        // Use creationURL() because url() can be changed via History.replaceState() so it's not reliable.
        if (equalIgnoringFragmentIdentifier(localFrame->document()->creationURL(), completeURL)) {
            if (foundOneSelfReference)
                return true;
            foundOneSelfReference = true;
        }
    }
    return false;
}

bool SubframeLoadingDisabler::canLoadFrame(HTMLFrameOwnerElement& owner)
{
    for (RefPtr<ContainerNode> node = &owner; node; node = node->parentOrShadowHostNode()) {
        if (disabledSubtreeRoots().contains(node.get()))
            return false;
    }
    return true;
}

} // namespace WebCore
