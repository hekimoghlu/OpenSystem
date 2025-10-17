/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "CSSParserContext.h"
#include "CSSParserToken.h"
#include "ContainerQuery.h"
#include "GenericMediaQueryParser.h"

namespace WebCore {
namespace CQ {

struct ContainerProgressProviding;

struct ContainerQueryParser : MQ::GenericMediaQueryParser<ContainerQueryParser>  {
    static std::optional<CQ::ContainerQuery> consumeContainerQuery(CSSParserTokenRange&, const MediaQueryParserContext&);

    static bool isValidFunctionId(CSSValueID);
    static const MQ::FeatureSchema* schemaForFeatureName(const AtomString&, const MediaQueryParserContext&, State&);
    static Vector<const MQ::FeatureSchema*> featureSchemas();

    // Accessor used by calc()'s container-progress() function to find a ContainerProgressProviding by name.
    static const ContainerProgressProviding* containerProgressProvidingSchemaForFeatureName(const AtomString&, const MediaQueryParserContext&);
};

}
}
