/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

#include "GenericMediaQueryParser.h"
#include "MediaQuery.h"
#include "MediaQueryParserContext.h"

namespace WebCore {
namespace MQ {

struct MediaProgressProviding;

struct MediaQueryParser : public GenericMediaQueryParser<MediaQueryParser>  {
    static MediaQueryList parse(const String&, const MediaQueryParserContext&);
    static MediaQueryList parse(CSSParserTokenRange, const MediaQueryParserContext&);
    static std::optional<MediaQuery> parseCondition(CSSParserTokenRange, const MediaQueryParserContext&);

    static MediaQueryList consumeMediaQueryList(CSSParserTokenRange&, const MediaQueryParserContext&);
    static std::optional<MediaQuery> consumeMediaQuery(CSSParserTokenRange&, const MediaQueryParserContext&);

    static const FeatureSchema* schemaForFeatureName(const AtomString&, const MediaQueryParserContext&, State&);
    static Vector<const FeatureSchema*> featureSchemas();

    // Accessor used by calc()'s media-progress() function to find a MediaProgressProviding by name.
    static const MediaProgressProviding* mediaProgressProvidingSchemaForFeatureName(const AtomString&, const MediaQueryParserContext&);
};

void serialize(StringBuilder&, const MediaQueryList&);
void serialize(StringBuilder&, const MediaQuery&);

}
}
