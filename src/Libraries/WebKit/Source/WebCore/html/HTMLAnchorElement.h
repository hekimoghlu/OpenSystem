/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

#include "Document.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "PrivateClickMeasurement.h"
#include "SharedStringHash.h"
#include "URLDecomposition.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class DOMTokenList;

enum class ReferrerPolicy : uint8_t;

// Link relation bitmask values.
enum class Relation : uint8_t {
    NoReferrer = 1 << 0,
    NoOpener = 1 << 1,
    Opener = 1 << 2,
};

class HTMLAnchorElement : public HTMLElement, public URLDecomposition {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLAnchorElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLAnchorElement);
public:
    static Ref<HTMLAnchorElement> create(Document&);
    static Ref<HTMLAnchorElement> create(const QualifiedName&, Document&);

    virtual ~HTMLAnchorElement();

    WEBCORE_EXPORT URL href() const;
    void setHref(const AtomString&);

    const AtomString& name() const;

    WEBCORE_EXPORT String origin() const;

    WEBCORE_EXPORT void setProtocol(StringView value);

    WEBCORE_EXPORT String text();
    void setText(String&&);

    bool isLiveLink() const;

    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    bool hasRel(Relation) const;
    
    inline SharedStringHash visitedLinkHash() const;

    WEBCORE_EXPORT DOMTokenList& relList();

#if USE(SYSTEM_PREVIEW)
    WEBCORE_EXPORT bool isSystemPreviewLink();
#endif

    void setReferrerPolicyForBindings(const AtomString&);
    String referrerPolicyForBindings() const;
    ReferrerPolicy referrerPolicy() const;

    Node::InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode& parentOfInsertedTree) override;

protected:
    HTMLAnchorElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;

private:
    bool supportsFocus() const override;
    bool isMouseFocusable() const override;
    bool isKeyboardFocusable(KeyboardEvent*) const override;
    void defaultEventHandler(Event&) final;
    void setActive(bool active, Style::InvalidationScope) final;
    bool isURLAttribute(const Attribute&) const final;
    bool canStartSelection() const final;
    AtomString target() const override;
    int defaultTabIndex() const final;
    bool draggable() const final;
    bool isInteractiveContent() const final;

    AtomString effectiveTarget() const;

    void sendPings(const URL& destinationURL);

    std::optional<URL> attributionDestinationURLForPCM() const;
    std::optional<RegistrableDomain> mainDocumentRegistrableDomainForPCM() const;
    std::optional<PCM::EphemeralNonce> attributionSourceNonceForPCM() const;
    std::optional<PrivateClickMeasurement> parsePrivateClickMeasurementForSKAdNetwork(const URL&) const;
    std::optional<PrivateClickMeasurement> parsePrivateClickMeasurement(const URL&) const;

    void handleClick(Event&);

    enum EventType {
        MouseEventWithoutShiftKey,
        MouseEventWithShiftKey,
        NonMouseEvent,
    };
    static EventType eventType(Event&);
    bool treatLinkAsLiveForEventType(EventType) const;

    Element* rootEditableElementForSelectionOnMouseDown() const;
    void setRootEditableElementForSelectionOnMouseDown(Element*);
    void clearRootEditableElementForSelectionOnMouseDown();

    URL fullURL() const final { return href(); }
    void setFullURL(const URL& fullURL) final { setHref(AtomString { fullURL.string() }); }

    bool m_hasRootEditableElementForSelectionOnMouseDown { false };
    bool m_wasShiftKeyDownOnMouseDown { false };
    OptionSet<Relation> m_linkRelations;

    // This is computed only once and must not be affected by subsequent URL changes.
    mutable Markable<SharedStringHash, SharedStringHashMarkableTraits> m_storedVisitedLinkHash;

    std::unique_ptr<DOMTokenList> m_relList;
};

// Functions shared with the other anchor elements (i.e., SVG).

bool isEnterKeyKeydownEvent(Event&);
bool shouldProhibitLinks(Element*);

} // namespace WebCore
