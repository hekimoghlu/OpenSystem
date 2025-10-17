/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#include "APIContentRuleList.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "WebCompiledContentRuleList.h"
#include <WebCore/CombinedURLFilters.h>
#include <WebCore/ContentExtensionParser.h>
#include <WebCore/URLFilterParser.h>

namespace API {

ContentRuleList::ContentRuleList(Ref<WebKit::WebCompiledContentRuleList>&& contentRuleList, WebKit::NetworkCache::Data&& mappedFile)
    : m_compiledRuleList(WTFMove(contentRuleList))
    , m_mappedFile(WTFMove(mappedFile))
{
}

ContentRuleList::~ContentRuleList()
{
}

const WTF::String& ContentRuleList::name() const
{
    return m_compiledRuleList->data().identifier;
}

bool ContentRuleList::supportsRegularExpression(const WTF::String& regex)
{
    using namespace WebCore::ContentExtensions;
    CombinedURLFilters combinedURLFilters;
    URLFilterParser urlFilterParser(combinedURLFilters);

    switch (urlFilterParser.addPattern(regex, false, 0)) {
    case URLFilterParser::Ok:
    case URLFilterParser::MatchesEverything:
        return true;
    case URLFilterParser::NonASCII:
    case URLFilterParser::UnsupportedCharacterClass:
    case URLFilterParser::BackReference:
    case URLFilterParser::ForwardReference:
    case URLFilterParser::MisplacedStartOfLine:
    case URLFilterParser::WordBoundary:
    case URLFilterParser::AtomCharacter:
    case URLFilterParser::Group:
    case URLFilterParser::Disjunction:
    case URLFilterParser::MisplacedEndOfLine:
    case URLFilterParser::EmptyPattern:
    case URLFilterParser::YarrError:
    case URLFilterParser::InvalidQuantifier:
        break;
    }
    return false;
}

std::error_code ContentRuleList::parseRuleList(const WTF::String& ruleList)
{
    auto result = WebCore::ContentExtensions::parseRuleList(ruleList);
    if (result.has_value())
        return { };

    return result.error();
}

} // namespace API

#endif // ENABLE(CONTENT_EXTENSIONS)
