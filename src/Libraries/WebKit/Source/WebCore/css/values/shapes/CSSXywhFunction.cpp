/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include "CSSXywhFunction.h"

#include "CSSPrimitiveNumericTypes+Serialization.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

void Serialize<Xywh>::operator()(StringBuilder& builder, const Xywh& value)
{
    // <xywh()> = xywh( <length-percentage>{2} <length-percentage [0,âˆž]>{2} [ round <'border-radius'> ]? )

    serializationForCSS(builder, value.location);
    builder.append(' ');
    serializationForCSS(builder, value.size);

    if (!hasDefaultValue(value.radii)) {
        builder.append(' ', nameLiteralForSerialization(CSSValueRound), ' ');
        serializationForCSS(builder, value.radii);
    }
}

} // namespace CSS
} // namespace WebCore
