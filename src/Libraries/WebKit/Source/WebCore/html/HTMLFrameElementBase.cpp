/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
#include "HTMLFrameElementBase.h"

#include "Document.h"
#include "DocumentInlines.h"
#include "ElementInlines.h"
#include "EventLoop.h"
#include "FocusController.h"
#include "FrameLoader.h"
#include "HTMLNames.h"
#include "JSDOMBindingSecurity.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "Quirks.h"
#include "RenderWidget.h"
#include "ScriptController.h"
#include "Settings.h"
#include "SubframeLoader.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFrameElementBase);

using namespace HTMLNames;

HTMLFrameElementBase::HTMLFrameElementBase(const QualifiedName& tagName, Document& document)
    : HTMLFrameOwnerElement(tagName, document, TypeFlag::HasCustomStyleResolveCallbacks)
{
}

bool HTMLFrameElementBase::canLoadScriptURL(const URL& scriptURL) const
{
    return canLoadURL(scriptURL);
}

bool HTMLFrameElementBase::canLoad() const
{
    // FIXME: Why is it valuable to return true when m_frameURL is empty?
    // FIXME: After openURL replaces an empty URL with the blank URL, this may no longer necessarily return true.
    return m_frameURL.isEmpty() || canLoadURL(m_frameURL);
}

bool HTMLFrameElementBase::canLoadURL(const String& relativeURL) const
{
    return canLoadURL(document().completeURL(relativeURL));
}

// Note that unlike HTMLPlugInImageElement::canLoadURL this uses ScriptController::canAccessFromCurrentOrigin.
bool HTMLFrameElementBase::canLoadURL(const URL& completeURL) const
{
    if (completeURL.protocolIsJavaScript()) {
        RefPtr<Document> contentDocument = this->contentDocument();
        if (contentDocument && !ScriptController::canAccessFromCurrentOrigin(contentDocument->frame(), document()))
            return false;
    }

    return !isProhibitedSelfReference(completeURL);
}

void HTMLFrameElementBase::openURL(LockHistory lockHistory, LockBackForwardList lockBackForwardList)
{
    if (!canLoad())
        return;

    if (m_frameURL.isEmpty())
        m_frameURL = AtomString { aboutBlankURL().string() };

    RefPtr parentFrame { document().frame() };
    if (!parentFrame)
        return;

    auto frameName = getNameAttribute();
    if (frameName.isNull() && UNLIKELY(document().settings().needsFrameNameFallbackToIdQuirk()))
        frameName = getIdAttribute();

    auto completeURL = document().completeURL(m_frameURL);
    auto finishOpeningURL = [this, weakThis = WeakPtr { *this }, frameName, lockHistory, lockBackForwardList, parentFrame = WTFMove(parentFrame), completeURL] {
        if (!weakThis)
            return;
        Ref protectedThis { *this };
        if (shouldLoadFrameLazily()) {
            parentFrame->loader().subframeLoader().createFrameIfNecessary(protectedThis.get(), frameName);
            return;
        }

        document().willLoadFrameElement(completeURL);
        parentFrame->loader().subframeLoader().requestFrame(*this, m_frameURL, frameName, lockHistory, lockBackForwardList);
    };

    document().quirks().triggerOptionalStorageAccessIframeQuirk(completeURL, WTFMove(finishOpeningURL));
}

void HTMLFrameElementBase::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    // FIXME: trimming whitespace is probably redundant with the URL parser
    if (name == srcdocAttr) {
        if (newValue.isNull())
            setLocation(attributeWithoutSynchronization(srcAttr).string().trim(isASCIIWhitespace));
        else
            setLocation(aboutSrcDocURL().string());
    } else if (name == srcAttr && !hasAttributeWithoutSynchronization(srcdocAttr))
        setLocation(newValue.string().trim(isASCIIWhitespace));
    else if (name == scrollingAttr && contentFrame())
        protectedContentFrame()->updateScrollingMode();
    else
        HTMLFrameOwnerElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

Node::InsertedIntoAncestorResult HTMLFrameElementBase::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLFrameOwnerElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
    return InsertedIntoAncestorResult::Done;
}

void HTMLFrameElementBase::didFinishInsertingNode()
{
    if (!isConnected())
        return;

    // DocumentFragments don't kick off any loads.
    if (!document().frame())
        return;

    if (!SubframeLoadingDisabler::canLoadFrame(*this))
        return;

    if (!renderer())
        invalidateStyleAndRenderersForSubtree();

    auto work = [this, weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;
        Ref<HTMLFrameElementBase> protectedThis { *this };
        m_openingURLAfterInserting = true;
        if (isConnected())
            openURL();
        m_openingURLAfterInserting = false;
    };
    if (!m_openingURLAfterInserting)
        work();
    else
        document().eventLoop().queueTask(TaskSource::DOMManipulation, WTFMove(work));
}

void HTMLFrameElementBase::didAttachRenderers()
{
    if (RenderWidget* part = renderWidget()) {
        if (RefPtr frame = contentFrame())
            part->setWidget(frame->virtualView());
    }
}

void HTMLFrameElementBase::setLocation(const String& str)
{
    if (document().settings().needsAcrobatFrameReloadingQuirk() && m_frameURL == str)
        return;

    if (!SubframeLoadingDisabler::canLoadFrame(*this))
        return;

    m_frameURL = AtomString(str);

    if (isConnected())
        openURL(LockHistory::No, LockBackForwardList::No);
}

void HTMLFrameElementBase::setLocation(JSC::JSGlobalObject& state, const String& newLocation)
{
    if (WTF::protocolIsJavaScript(newLocation)) {
        if (!BindingSecurity::shouldAllowAccessToNode(state, contentDocument()))
            return;
    }

    setLocation(newLocation);
}

bool HTMLFrameElementBase::supportsFocus() const
{
    return true;
}

void HTMLFrameElementBase::setFocus(bool received, FocusVisibility visibility)
{
    HTMLFrameOwnerElement::setFocus(received, visibility);
    if (Page* page = document().page()) {
        CheckedRef focusController { page->focusController() };
        if (received)
            focusController->setFocusedFrame(contentFrame());
        else if (focusController->focusedFrame() == contentFrame()) // Focus may have already been given to another frame, don't take it away.
            focusController->setFocusedFrame(nullptr);
    }
}

bool HTMLFrameElementBase::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == srcAttr || attribute.name() == longdescAttr || HTMLFrameOwnerElement::isURLAttribute(attribute);
}

bool HTMLFrameElementBase::isHTMLContentAttribute(const Attribute& attribute) const
{
    return attribute.name() == srcdocAttr || HTMLFrameOwnerElement::isHTMLContentAttribute(attribute);
}

ScrollbarMode HTMLFrameElementBase::scrollingMode() const
{
    auto scrollingAttribute = attributeWithoutSynchronization(scrollingAttr);
    return equalLettersIgnoringASCIICase(scrollingAttribute, "no"_s)
        || equalLettersIgnoringASCIICase(scrollingAttribute, "noscroll"_s)
        || equalLettersIgnoringASCIICase(scrollingAttribute, "off"_s)
        ? ScrollbarMode::AlwaysOff : ScrollbarMode::Auto;
}

} // namespace WebCore
