/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include "MediaList.h"

#include "CSSImportRule.h"
#include "CSSMediaRule.h"
#include "CSSStyleSheet.h"
#include "Document.h"
#include "LocalDOMWindow.h"
#include "MediaQuery.h"
#include "MediaQueryParser.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

MediaList::MediaList(CSSStyleSheet* parentSheet)
    : m_parentStyleSheet(parentSheet)
{
}

MediaList::MediaList(CSSRule* parentRule)
    : m_parentRule(parentRule)
{
}

MediaList::~MediaList() = default;

void MediaList::detachFromParent()
{
    m_detachedMediaQueries = mediaQueries();
    m_parentStyleSheet = nullptr;
    m_parentRule = nullptr;
}

unsigned MediaList::length() const
{
    return mediaQueries().size();
}

const MQ::MediaQueryList& MediaList::mediaQueries() const
{
    if (m_detachedMediaQueries)
        return *m_detachedMediaQueries;
    if (auto* rule = dynamicDowncast<CSSImportRule>(m_parentRule))
        return rule->mediaQueries();
    if (auto* rule = dynamicDowncast<CSSMediaRule>(m_parentRule))
        return rule->mediaQueries();
    return m_parentStyleSheet->mediaQueries();
}

void MediaList::setMediaQueries(MQ::MediaQueryList&& queries)
{
    if (m_parentStyleSheet) {
        m_parentStyleSheet->setMediaQueries(WTFMove(queries));
        m_parentStyleSheet->didMutate();
        return;
    }

    CSSStyleSheet::RuleMutationScope mutationScope(m_parentRule);
    if (auto* rule = dynamicDowncast<CSSImportRule>(m_parentRule))
        rule->setMediaQueries(WTFMove(queries));
    if (auto* rule = dynamicDowncast<CSSMediaRule>(m_parentRule))
        rule->setMediaQueries(WTFMove(queries));
}

String MediaList::mediaText() const
{
    StringBuilder builder;
    MQ::serialize(builder, mediaQueries());
    return builder.toString();
}

void MediaList::setMediaText(const String& value)
{
    setMediaQueries(MQ::MediaQueryParser::parse(value, { }));
}

String MediaList::item(unsigned index) const
{
    auto& queries = mediaQueries();
    if (index < queries.size()) {
        StringBuilder builder;
        MQ::serialize(builder, queries[index]);
        return builder.toString();
    }
    return { };
}

ExceptionOr<void> MediaList::deleteMedium(const String& value)
{
    auto valueToRemove = value.convertToASCIILowercase();
    auto queries = mediaQueries();
    for (unsigned i = 0; i < queries.size(); ++i) {
        if (item(i) == valueToRemove) {
            queries.remove(i);
            setMediaQueries(WTFMove(queries));
            return { };
        }
    }
    return Exception { ExceptionCode::NotFoundError };
}

void MediaList::appendMedium(const String& value)
{
    if (value.isEmpty())
        return;

    auto newQuery = MQ::MediaQueryParser::parse(value, { });

    auto queries = mediaQueries();
    queries.appendVector(newQuery);
    setMediaQueries(WTFMove(queries));
}

TextStream& operator<<(TextStream& ts, const MediaList& mediaList)
{
    ts << mediaList.mediaText();
    return ts;
}

} // namespace WebCore

