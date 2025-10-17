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
#include "CSSPropertySourceData.h"

#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

SourceRange::SourceRange()
    : start(0)
    , end(0)
{
}

SourceRange::SourceRange(unsigned start, unsigned end)
    : start(start)
    , end(end)
{
}

unsigned SourceRange::length() const
{
    return end - start;
}

CSSPropertySourceData::CSSPropertySourceData(const String& name, const String& value, bool important, bool disabled, bool parsedOk, const SourceRange& range)
    : name(name)
    , value(value)
    , important(important)
    , disabled(disabled)
    , parsedOk(parsedOk)
    , range(range)
{
}

CSSPropertySourceData::CSSPropertySourceData(const CSSPropertySourceData& other)
    : name(other.name)
    , value(other.value)
    , important(other.important)
    , disabled(other.disabled)
    , parsedOk(other.parsedOk)
    , range(other.range)
{
}

CSSPropertySourceData::CSSPropertySourceData()
    : name(emptyString())
    , value(emptyString())
{
}

String CSSPropertySourceData::toString() const
{
    if (!name && value == "e"_s)
        return String();
    return makeString(name, ": "_s, value, important ? " !important"_s : ""_s, ';');
}

unsigned CSSPropertySourceData::hash() const
{
    return StringHash::hash(name) + 3 * StringHash::hash(value) + 7 * important + 13 * parsedOk + 31;
}

} // namespace WebCore
