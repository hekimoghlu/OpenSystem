/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace RFC8941 {

class Token {
public:
    explicit Token(String&& token) : m_token(WTFMove(token)) { }
    const String& string() const { return m_token; }
private:
    String m_token;
};

using BareItem = std::variant<String, Token, bool>; // FIXME: The specification supports more BareItem types.

class Parameters {
public:
    Parameters() = default;
    explicit Parameters(HashMap<String, BareItem>&& parameters)
        : m_parameters(WTFMove(parameters)) { }
    const HashMap<String, BareItem>& map() const { return m_parameters; }
    template<typename T> const T* getIf(ASCIILiteral key) const;
private:
    HashMap<String, BareItem> m_parameters;
};

template<typename T> const T* Parameters::getIf(ASCIILiteral key) const
{
    auto it = m_parameters.find<HashTranslatorASCIILiteral>(key);
    if (it == m_parameters.end())
        return nullptr;
    return std::get_if<T>(&(it->value));
}

using InnerList = Vector<std::pair<BareItem, Parameters>>;
using ItemOrInnerList = std::variant<BareItem, InnerList>;

WEBCORE_EXPORT std::optional<std::pair<BareItem, Parameters>> parseItemStructuredFieldValue(StringView header);
WEBCORE_EXPORT std::optional<HashMap<String, std::pair<ItemOrInnerList, Parameters>>> parseDictionaryStructuredFieldValue(StringView header);

} // namespace RFC8941

