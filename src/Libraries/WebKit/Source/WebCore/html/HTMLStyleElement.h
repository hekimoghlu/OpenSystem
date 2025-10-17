/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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

#include "HTMLElement.h"
#include "InlineStyleSheetOwner.h"

namespace WebCore {

class HTMLStyleElement;
class Page;
class StyleSheet;

template<typename T, typename Counter> class EventSender;

using StyleEventSender = EventSender<HTMLStyleElement, WeakPtrImplWithEventTargetData>;

class HTMLStyleElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLStyleElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLStyleElement);
public:
    static Ref<HTMLStyleElement> create(Document&);
    static Ref<HTMLStyleElement> create(const QualifiedName&, Document&, bool createdByParser);
    virtual ~HTMLStyleElement();

    CSSStyleSheet* sheet() const { return m_styleSheetOwner.sheet(); }

    WEBCORE_EXPORT bool disabled() const;
    WEBCORE_EXPORT void setDisabled(bool);

    void dispatchPendingEvent(StyleEventSender*, const AtomString& eventType);
    static void dispatchPendingLoadEvents(Page*);

    void finishParsingChildren() final;

    WEBCORE_EXPORT DOMTokenList& blocking();

private:
    HTMLStyleElement(const QualifiedName&, Document&, bool createdByParser);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void childrenChanged(const ChildChange&) final;

    bool isLoading() const { return m_styleSheetOwner.isLoading(); }
    bool sheetLoaded() final { return m_styleSheetOwner.sheetLoaded(*this); }
    void notifyLoadedSheetAndAllCriticalSubresources(bool errorOccurred) final;
    void startLoadingDynamicSheet() final { m_styleSheetOwner.startLoadingDynamicSheet(*this); }

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    InlineStyleSheetOwner m_styleSheetOwner;
    std::unique_ptr<DOMTokenList> m_blockingList;
    bool m_loadedSheet { false };
};

} //namespace
