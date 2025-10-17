/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "DOMTokenList.h"

#include "SpaceSplitString.h"
#include <wtf/HashSet.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMTokenList);

DOMTokenList::DOMTokenList(Element& element, const QualifiedName& attributeName, IsSupportedTokenFunction&& isSupportedToken)
    : m_element(element)
    , m_attributeName(attributeName)
    , m_isSupportedToken(WTFMove(isSupportedToken))
{
}

static inline bool tokenContainsHTMLSpace(StringView token)
{
    return token.find(isASCIIWhitespace<UChar>) != notFound;
}

ExceptionOr<void> DOMTokenList::validateToken(StringView token)
{
    if (token.isEmpty())
        return Exception { ExceptionCode::SyntaxError };

    if (tokenContainsHTMLSpace(token))
        return Exception { ExceptionCode::InvalidCharacterError };

    return { };
}

ExceptionOr<void> DOMTokenList::validateTokens(std::span<const AtomString> tokens)
{
    for (auto& token : tokens) {
        auto result = validateToken(token);
        if (result.hasException())
            return result;
    }
    return { };
}

bool DOMTokenList::contains(const AtomString& token) const
{
    return tokens().contains(token);
}

inline ExceptionOr<void> DOMTokenList::addInternal(std::span<const AtomString> newTokens)
{
    // This is usually called with a single token.
    Vector<AtomString, 1> uniqueNewTokens;
    uniqueNewTokens.reserveInitialCapacity(newTokens.size());

    auto& tokens = this->tokens();

    for (auto& newToken : newTokens) {
        auto result = validateToken(newToken);
        if (result.hasException())
            return result;
        if (!tokens.contains(newToken) && !uniqueNewTokens.contains(newToken))
            uniqueNewTokens.append(newToken);
    }

    if (!uniqueNewTokens.isEmpty())
        tokens.appendVector(uniqueNewTokens);

    updateAssociatedAttributeFromTokens();

    return { };
}

ExceptionOr<void> DOMTokenList::add(const FixedVector<AtomString>& tokens)
{
    return addInternal(tokens);
}

ExceptionOr<void> DOMTokenList::add(const AtomString& token)
{
    return addInternal({ &token, 1 });
}

inline ExceptionOr<void> DOMTokenList::removeInternal(std::span<const AtomString> tokensToRemove)
{
    auto result = validateTokens(tokensToRemove);
    if (result.hasException())
        return result;

    auto& tokens = this->tokens();
    for (auto& tokenToRemove : tokensToRemove)
        tokens.removeFirst(tokenToRemove);

    updateAssociatedAttributeFromTokens();

    return { };
}

ExceptionOr<void> DOMTokenList::remove(const FixedVector<AtomString>& tokens)
{
    return removeInternal(tokens);
}

ExceptionOr<void> DOMTokenList::remove(const AtomString& token)
{
    return removeInternal({ &token, 1 });
}

ExceptionOr<bool> DOMTokenList::toggle(const AtomString& token, std::optional<bool> force)
{
    auto result = validateToken(token);
    if (result.hasException())
        return result.releaseException();

    auto& tokens = this->tokens();

    if (tokens.contains(token)) {
        if (!force.value_or(false)) {
            tokens.removeFirst(token);
            updateAssociatedAttributeFromTokens();
            return false;
        }
        return true;
    }

    if (force && !force.value())
        return false;

    tokens.append(token);
    updateAssociatedAttributeFromTokens();
    return true;
}

static inline void replaceInOrderedSet(Vector<AtomString, 1>& tokens, size_t tokenIndex, const AtomString& newToken)
{
    ASSERT(tokenIndex != notFound);
    ASSERT(tokenIndex < tokens.size());

    auto newTokenIndex = tokens.find(newToken);
    if (newTokenIndex == notFound) {
        tokens[tokenIndex] = newToken;
        return;
    }

    if (newTokenIndex == tokenIndex)
        return;

    if (newTokenIndex > tokenIndex) {
        tokens[tokenIndex] = newToken;
        tokens.remove(newTokenIndex);
    } else
        tokens.remove(tokenIndex);
}

// https://dom.spec.whatwg.org/#dom-domtokenlist-replace
ExceptionOr<bool> DOMTokenList::replace(const AtomString& token, const AtomString& newToken)
{
    if (token.isEmpty() || newToken.isEmpty())
        return Exception { ExceptionCode::SyntaxError };

    if (tokenContainsHTMLSpace(token) || tokenContainsHTMLSpace(newToken))
        return Exception { ExceptionCode::InvalidCharacterError };

    auto& tokens = this->tokens();

    auto tokenIndex = tokens.find(token);
    if (tokenIndex == notFound)
        return false;

    replaceInOrderedSet(tokens, tokenIndex, newToken);
    ASSERT(token == newToken || tokens.find(token) == notFound);

    updateAssociatedAttributeFromTokens();

    return true;
}

// https://dom.spec.whatwg.org/#concept-domtokenlist-validation
ExceptionOr<bool> DOMTokenList::supports(StringView token)
{
    if (!m_isSupportedToken)
        return Exception { ExceptionCode::TypeError };
    return m_isSupportedToken(m_element->document(), token);
}

// https://dom.spec.whatwg.org/#dom-domtokenlist-value
const AtomString& DOMTokenList::value() const
{
    return protectedElement()->getAttribute(m_attributeName);
}

void DOMTokenList::setValue(const AtomString& value)
{
    protectedElement()->setAttribute(m_attributeName, value);
}

void DOMTokenList::updateTokensFromAttributeValue(const AtomString& value)
{
    // Clear tokens but not capacity.
    m_tokens.shrink(0);

    UncheckedKeyHashSet<AtomString> addedTokens;
    // https://dom.spec.whatwg.org/#ordered%20sets
    for (unsigned start = 0; ; ) {
        while (start < value.length() && isASCIIWhitespace(value[start]))
            ++start;
        if (start >= value.length())
            break;
        unsigned end = start + 1;
        while (end < value.length() && !isASCIIWhitespace(value[end]))
            ++end;
        bool wholeAttributeIsSingleToken = !start && end == value.length();
        if (wholeAttributeIsSingleToken) {
            m_tokens.append(value);
            break;
        }

        auto tokenView = StringView { value }.substring(start, end - start);
        if (!addedTokens.contains<StringViewHashTranslator>(tokenView)) {
            auto token = tokenView.toAtomString();
            m_tokens.append(token);
            addedTokens.add(WTFMove(token));
        }

        start = end + 1;
    }

    m_tokens.shrinkToFit();
    m_tokensNeedUpdating = false;
}

// https://dom.spec.whatwg.org/#concept-dtl-update
void DOMTokenList::updateAssociatedAttributeFromTokens()
{
    ASSERT(!m_tokensNeedUpdating);

    Ref element = m_element.get();
    if (m_tokens.isEmpty() && !element->hasAttribute(m_attributeName))
        return;

    if (m_tokens.isEmpty()) {
        element->setAttribute(m_attributeName, emptyAtom());
        return;
    }

    bool wholeAttributeIsSingleToken = m_tokens.size() == 1;
    if (wholeAttributeIsSingleToken) {
        SetForScope inAttributeUpdate(m_inUpdateAssociatedAttributeFromTokens, true);
        element->setAttribute(m_attributeName, m_tokens[0]);
        return;
    }

    // https://dom.spec.whatwg.org/#concept-ordered-set-serializer
    StringBuilder builder;
    for (auto& token : tokens()) {
        if (!builder.isEmpty())
            builder.append(' ');
        builder.append(token);
    }
    AtomString serializedValue = builder.toAtomString();

    SetForScope inAttributeUpdate(m_inUpdateAssociatedAttributeFromTokens, true);
    element->setAttribute(m_attributeName, serializedValue);
}

Vector<AtomString, 1>& DOMTokenList::tokens()
{
    if (m_tokensNeedUpdating)
        updateTokensFromAttributeValue(protectedElement()->getAttribute(m_attributeName));
    ASSERT(!m_tokensNeedUpdating);
    return m_tokens;
}

} // namespace WebCore
