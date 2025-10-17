/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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

#include "CSSStyleSheet.h"
#include "CachedStyleSheetClient.h"
#include "CachedResourceHandle.h"
#include "DOMTokenList.h"
#include "HTMLElement.h"
#include "LinkLoader.h"
#include "LinkLoaderClient.h"
#include "LinkRelAttribute.h"

namespace WebCore {

class DOMTokenList;
class ExpectIdTargetObserver;
class HTMLLinkElement;
class Page;
struct MediaQueryParserContext;

enum class RequestPriority : uint8_t;

template<typename T, typename Counter> class EventSender;
using LinkEventSender = EventSender<HTMLLinkElement, WeakPtrImplWithEventTargetData>;

class HTMLLinkElement final : public HTMLElement, public CachedStyleSheetClient, public LinkLoaderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLLinkElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLLinkElement);
public:
    USING_CAN_MAKE_WEAKPTR(HTMLElement);

    static Ref<HTMLLinkElement> create(const QualifiedName&, Document&, bool createdByParser);
    virtual ~HTMLLinkElement();

    URL href() const;
    WEBCORE_EXPORT const AtomString& rel() const;

    AtomString target() const final;

    const AtomString& type() const;

    std::optional<LinkIconType> iconType() const;

    CSSStyleSheet* sheet() const { return m_sheet.get(); }

    bool styleSheetIsLoading() const;

    bool isDisabled() const { return m_disabledState == Disabled; }
    bool isEnabledViaScript() const { return m_disabledState == EnabledViaScript; }
    DOMTokenList& sizes();

    WEBCORE_EXPORT bool mediaAttributeMatches() const;

    WEBCORE_EXPORT void setCrossOrigin(const AtomString&);
    WEBCORE_EXPORT String crossOrigin() const;
    WEBCORE_EXPORT void setAs(const AtomString&);
    WEBCORE_EXPORT String as() const;

    void dispatchPendingEvent(LinkEventSender*, const AtomString& eventType);
    static void dispatchPendingLoadEvents(Page*);

    WEBCORE_EXPORT DOMTokenList& relList();
    WEBCORE_EXPORT DOMTokenList& blocking();

#if ENABLE(APPLICATION_MANIFEST)
    bool isApplicationManifest() const { return m_relAttribute.isApplicationManifest; }
#endif

    void allowPrefetchLoadAndErrorForTesting() { m_allowPrefetchLoadAndErrorForTesting = true; }

    void setReferrerPolicyForBindings(const AtomString&);
    String referrerPolicyForBindings() const;
    ReferrerPolicy referrerPolicy() const;

    void setFetchPriorityForBindings(const AtomString&);
    String fetchPriorityForBindings() const;
    RequestPriority fetchPriority() const;

    // If element is specified, checks if that Element satisfies the link.
    // Otherwise checks if any Element in the tree does.
    void processInternalResourceLink(Element* = nullptr);

private:
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    bool shouldLoadLink() final;
    void process();
    static void processCallback(Node*);
    void clearSheet();

    void potentiallyBlockRendering();
    void unblockRendering();
    bool isImplicitlyPotentiallyRenderBlocking() const;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    void initializeStyleSheet(Ref<StyleSheetContents>&&, const CachedCSSStyleSheet&, MediaQueryParserContext);

    // from CachedResourceClient
    void setCSSStyleSheet(const String& href, const URL& baseURL, ASCIILiteral charset, const CachedCSSStyleSheet*) final;
    bool sheetLoaded() final;
    void notifyLoadedSheetAndAllCriticalSubresources(bool errorOccurred) final;
    void startLoadingDynamicSheet() final;

    void linkLoaded() final;
    void linkLoadingErrored() final;

    bool isAlternate() const { return m_disabledState == Unset && m_relAttribute.isAlternate; }
    
    void setDisabledState(bool);

    bool isURLAttribute(const Attribute&) const final;

    HTMLLinkElement(const QualifiedName&, Document&, bool createdByParser);

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    void finishParsingChildren() final;

    String debugDescription() const final;

    enum class PendingSheetType : uint8_t { Unknown, Active, Inactive };
    void addPendingSheet(PendingSheetType);

    void removePendingSheet();

    LinkLoader m_linkLoader;
    Style::Scope* m_styleScope { nullptr };
    CachedResourceHandle<CachedCSSStyleSheet> m_cachedSheet;
    RefPtr<CSSStyleSheet> m_sheet;
    enum DisabledState : uint8_t {
        Unset,
        EnabledViaScript,
        Disabled
    };

    String m_type;
    String m_media;
    String m_integrityMetadataForPendingSheetRequest;
    URL m_url;
    std::unique_ptr<DOMTokenList> m_sizes;
    std::unique_ptr<DOMTokenList> m_relList;
    std::unique_ptr<DOMTokenList> m_blockingList;
    std::unique_ptr<ExpectIdTargetObserver> m_expectIdTargetObserver;
    DisabledState m_disabledState;
    LinkRelAttribute m_relAttribute;
    bool m_loading : 1;
    bool m_createdByParser : 1;
    bool m_loadedResource : 1;
    bool m_isHandlingBeforeLoad : 1;
    bool m_allowPrefetchLoadAndErrorForTesting : 1;
    bool m_isRenderBlocking : 1 { false };
    PendingSheetType m_pendingSheetType;
};

}
