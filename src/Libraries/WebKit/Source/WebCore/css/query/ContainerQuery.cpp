/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#include "ContainerQuery.h"

#include "CSSMarkup.h"
#include "CSSValue.h"
#include "ContainerQueryFeatures.h"
#include "GenericMediaQuerySerialization.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CQ {

OptionSet<Axis> requiredAxesForFeature(const MQ::Feature& feature)
{
    if (feature.schema == &Features::width())
        return { Axis::Width };
    if (feature.schema == &Features::height())
        return { Axis::Height };
    if (feature.schema == &Features::inlineSize())
        return { Axis::Inline };
    if (feature.schema == &Features::blockSize())
        return { Axis::Block };
    if (feature.schema == &Features::aspectRatio() || feature.schema == &Features::orientation())
        return { Axis::Inline, Axis::Block };
    return { };
}

void serialize(StringBuilder& builder, const ContainerQuery& query)
{
    auto name = query.name;
    if (!name.isEmpty()) {
        serializeIdentifier(name, builder);
        builder.append(' ');
    }

    serialize(builder, query.condition);
}

}
}

