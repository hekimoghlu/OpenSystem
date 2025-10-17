/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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

#include "Element.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DOMTokenList final {
    WTF_MAKE_TZONE_ALLOCATED(DOMTokenList);
public:
    using IsSupportedTokenFunction = Function<bool(Document&, StringView)>;
    DOMTokenList(Element&, const QualifiedName& attributeName, IsSupportedTokenFunction&& isSupportedToken = { });

    inline void associatedAttributeValueChanged();

    void ref() { m_element->ref(); }
    void deref() { m_element->deref(); }

    unsigned length() const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    const AtomString& item(unsigned index) const;

    WEBCORE_EXPORT bool contains(const AtomString&) const;
    ExceptionOr<void> add(const FixedVector<AtomString>&);
    ExceptionOr<void> add(const AtomString&);
    ExceptionOr<void> remove(const FixedVector<AtomString>&);
    ExceptionOr<void> remove(const AtomString&);
    WEBCORE_EXPORT ExceptionOr<bool> toggle(const AtomString&, std::optional<bool> force);
    ExceptionOr<bool> replace(const AtomString& token, const AtomString& newToken);
    ExceptionOr<bool> supports(StringView token);

    Element& element() const { return m_element.get(); }
    Ref<Element> protectedElement() const { return m_element.get(); }

    WEBCORE_EXPORT void setValue(const AtomString&);
    WEBCORE_EXPORT const AtomString& value() const;

private:
    void updateTokensFromAttributeValue(const AtomString&);
    void updateAssociatedAttributeFromTokens();

    WEBCORE_EXPORT Vector<AtomString, 1>& tokens();
    const Vector<AtomString, 1>& tokens() const { return const_cast<DOMTokenList&>(*this).tokens(); }

    static ExceptionOr<void> validateToken(StringView);
    static ExceptionOr<void> validateTokens(std::span<const AtomString> tokens);
    ExceptionOr<void> addInternal(std::span<const AtomString> tokens);
    ExceptionOr<void> removeInternal(std::span<const AtomString> tokens);

    CheckedRef<Element> m_element;
    const WebCore::QualifiedName& m_attributeName;
    bool m_inUpdateAssociatedAttributeFromTokens { false };
    bool m_tokensNeedUpdating { true };
    Vector<AtomString, 1> m_tokens;
    IsSupportedTokenFunction m_isSupportedToken;
};

inline unsigned DOMTokenList::length() const
{
    return tokens().size();
}

inline const AtomString& DOMTokenList::item(unsigned index) const
{
    auto& tokens = this->tokens();
    return index < tokens.size() ? tokens[index] : nullAtom();
}

inline void DOMTokenList::associatedAttributeValueChanged()
{
    // Do not reset the DOMTokenList value if the attribute value was changed by us.
    if (m_inUpdateAssociatedAttributeFromTokens)
        return;
    m_tokensNeedUpdating = true;
}

} // namespace WebCore
