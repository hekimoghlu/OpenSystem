/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "CSSRayFunction.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"

namespace WebCore {
namespace CSS {

void Serialize<Ray>::operator()(StringBuilder& builder, const Ray& value)
{
    // ray() = ray( <angle> && <ray-size>? && contain? && [at <position>]? )
    // https://drafts.fxtf.org/motion-1/#ray-function

    serializationForCSS(builder, value.angle);

    if (!std::holds_alternative<Keyword::ClosestSide>(value.size)) {
        builder.append(' ');
        serializationForCSS(builder, value.size);
    }

    if (value.contain) {
        builder.append(' ');
        serializationForCSS(builder, *value.contain);
    }

    if (value.position) {
        builder.append(' ', nameLiteralForSerialization(CSSValueAt), ' ');
        serializationForCSS(builder, *value.position);
    }
}

} // namespace CSS
} // namespace WebCore
