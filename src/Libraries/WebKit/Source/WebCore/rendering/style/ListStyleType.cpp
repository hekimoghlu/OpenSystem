/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#include "ListStyleType.h"

#include "CSSPrimitiveValueMappings.h"
#include "CSSValueKeywords.h"

namespace WebCore {

bool ListStyleType::isCircle() const
{
    return type == Type::CounterStyle && identifier == nameLiteral(CSSValueCircle);
}

bool ListStyleType::isSquare() const
{
    return type == Type::CounterStyle && identifier == nameLiteral(CSSValueSquare);
}

bool ListStyleType::isDisc() const
{
    return type == Type::CounterStyle && identifier == nameLiteral(CSSValueDisc);
}

TextStream& operator<<(TextStream& ts, ListStyleType::Type styleType)
{
    return ts << nameLiteral(toCSSValueID(styleType)).characters();
}

WTF::TextStream& operator<<(WTF::TextStream& ts, ListStyleType listStyle)
{
    if (listStyle.type == ListStyleType::Type::CounterStyle)
        ts << listStyle.identifier;
    else if (listStyle.type == ListStyleType::Type::String)
        ts << "\"" << listStyle.identifier << "\"";
    else
        ts << listStyle.type;
    return ts;
}

} // namespace WebCore
