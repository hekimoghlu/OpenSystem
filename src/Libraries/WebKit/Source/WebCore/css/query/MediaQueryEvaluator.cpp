/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#include "MediaQueryEvaluator.h"

#include "CSSToLengthConversionData.h"
#include "Document.h"
#include "FontCascade.h"
#include "MediaQuery.h"
#include "MediaQueryFeatures.h"
#include "RenderView.h"
#include "StyleFontSizeFunctions.h"

namespace WebCore {
namespace MQ {

MediaQueryEvaluator::MediaQueryEvaluator(const AtomString& mediaType, const Document& document, const RenderStyle* rootElementStyle)
    : GenericMediaQueryEvaluator()
    , m_mediaType(mediaType)
    , m_document(document)
    , m_rootElementStyle(rootElementStyle)
{
}

MediaQueryEvaluator::MediaQueryEvaluator(const AtomString& mediaType, EvaluationResult mediaConditionResult)
    : GenericMediaQueryEvaluator()
    , m_mediaType(mediaType)
    , m_staticMediaConditionResult(mediaConditionResult)
{
}

bool MediaQueryEvaluator::evaluate(const MediaQueryList& list) const
{
    if (list.isEmpty())
        return true;

    for (auto& query : list) {
        if (evaluate(query))
            return true;
    }
    return false;
}

bool MediaQueryEvaluator::evaluate(const MediaQuery& query) const
{
    bool isNegated = query.prefix && *query.prefix == Prefix::Not;

    if (!evaluateMediaType(query))
        return isNegated;

    auto result = [&] {
        if (!query.condition)
            return EvaluationResult::True;

        RefPtr document = m_document.get();
        if (!document || !m_rootElementStyle)
            return m_staticMediaConditionResult;

        if (!document->view() || !document->documentElement())
            return EvaluationResult::Unknown;

        auto defaultStyle = RenderStyle::create();
        auto fontDescription = defaultStyle.fontDescription();
        auto size = Style::fontSizeForKeyword(CSSValueMedium, false, *document);
        fontDescription.setComputedSize(size);
        fontDescription.setSpecifiedSize(size);
        defaultStyle.setFontDescription(WTFMove(fontDescription));

        FeatureEvaluationContext context { *document, { *m_rootElementStyle, &defaultStyle, nullptr, document->renderView() }, nullptr };
        return evaluateCondition(*query.condition, context);
    }();

    switch (result) {
    case EvaluationResult::Unknown:
        return false;
    case EvaluationResult::True:
        return !isNegated;
    case EvaluationResult::False:
        return isNegated;
    }

    ASSERT_NOT_REACHED();
    return false;
}

bool MediaQueryEvaluator::evaluateMediaType(const MediaQuery& query) const
{
    if (query.mediaType.isEmpty())
        return true;
    if (query.mediaType == allAtom())
        return true;
    return query.mediaType == m_mediaType;
};

OptionSet<MediaQueryDynamicDependency> MediaQueryEvaluator::collectDynamicDependencies(const MediaQueryList& queries) const
{
    OptionSet<MediaQueryDynamicDependency> result;

    for (auto& query : queries)
        result.add(collectDynamicDependencies(query));

    return result;
}

OptionSet<MediaQueryDynamicDependency> MediaQueryEvaluator::collectDynamicDependencies(const MediaQuery& query) const
{
    if (!evaluateMediaType(query))
        return { };

    OptionSet<MediaQueryDynamicDependency> result;

    traverseFeatures(query, [&](const Feature& feature) {
        if (!feature.schema)
            return;
        result.add(feature.schema->dependencies);
    });

    return result;
}

bool MediaQueryEvaluator::isPrintMedia() const
{
    return m_mediaType == printAtom();
}

}
}
