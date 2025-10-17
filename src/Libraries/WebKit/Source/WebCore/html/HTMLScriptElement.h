/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#include "DOMTokenList.h"
#include "HTMLElement.h"
#include "ScriptElement.h"

namespace WebCore {

class TrustedScript;
class TrustedScriptURL;

enum class RequestPriority : uint8_t;

class HTMLScriptElement final : public HTMLElement, public ScriptElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLScriptElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLScriptElement);
public:
    static Ref<HTMLScriptElement> create(const QualifiedName&, Document&, bool wasInsertedByParser, bool alreadyStarted = false);

    String text() const { return scriptContent(); }
    WEBCORE_EXPORT void setText(String&&);
    ExceptionOr<void> setText(std::variant<RefPtr<TrustedScript>, String>&&);

    using Node::setTextContent;
    ExceptionOr<void> setTextContent(std::optional<std::variant<RefPtr<TrustedScript>, String>>&&);

    using HTMLElement::setInnerText;
    ExceptionOr<void> setInnerText(std::variant<RefPtr<TrustedScript>, String>&&);

    String src() const;
    ExceptionOr<void> setSrc(std::variant<RefPtr<TrustedScriptURL>, String>&&);

    WEBCORE_EXPORT void setAsync(bool);
    WEBCORE_EXPORT bool async() const;

    WEBCORE_EXPORT void setCrossOrigin(const AtomString&);
    WEBCORE_EXPORT String crossOrigin() const;

    void setReferrerPolicyForBindings(const AtomString&);
    String referrerPolicyForBindings() const;
    ReferrerPolicy referrerPolicy() const final;

    using HTMLElement::ref;
    using HTMLElement::deref;

    static bool supports(StringView type) { return type == "classic"_s || type == "module"_s || type == "importmap"_s; }

    void setFetchPriorityForBindings(const AtomString&);
    String fetchPriorityForBindings() const;
    RequestPriority fetchPriority() const override;

    WEBCORE_EXPORT DOMTokenList& blocking();

private:
    HTMLScriptElement(const QualifiedName&, Document&, bool wasInsertedByParser, bool alreadyStarted);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void childrenChanged(const ChildChange&) final;
    void finishParsingChildren() final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    void potentiallyBlockRendering() final;
    void unblockRendering() final;
    bool isImplicitlyPotentiallyRenderBlocking() const;

    ExceptionOr<void> setTextContent(ExceptionOr<String>);

    bool isURLAttribute(const Attribute&) const final;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    String sourceAttributeValue() const final;
    AtomString charsetAttributeValue() const final;
    String typeAttributeValue() const final;
    String languageAttributeValue() const final;
    bool hasAsyncAttribute() const final;
    bool hasDeferAttribute() const final;
    bool hasNoModuleAttribute() const final;
    bool hasSourceAttribute() const final;

    void dispatchLoadEvent() final;

    bool isScriptPreventedByAttributes() const final;

    Ref<Element> cloneElementWithoutAttributesAndChildren(TreeScope&) final;

    std::unique_ptr<DOMTokenList> m_blockingList;
    bool m_isRenderBlocking { false };
};

} //namespace
