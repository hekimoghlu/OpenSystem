/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#include "CSSKeywordValue.h"

#include "CSSMarkup.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParser.h"
#include "CSSValuePool.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSKeywordValue);

Ref<CSSKeywordValue> CSSKeywordValue::rectifyKeywordish(CSSKeywordish&& keywordish)
{
    // https://drafts.css-houdini.org/css-typed-om/#rectify-a-keywordish-value
    return WTF::switchOn(WTFMove(keywordish), [] (String string) {
        return adoptRef(*new CSSKeywordValue(string));
    }, [] (RefPtr<CSSKeywordValue> value) {
        RELEASE_ASSERT(value);
        return value.releaseNonNull();
    });
}

ExceptionOr<Ref<CSSKeywordValue>> CSSKeywordValue::create(const String& value)
{
    if (value.isEmpty())
        return Exception { ExceptionCode::TypeError };
    
    return adoptRef(*new CSSKeywordValue(value));
}

ExceptionOr<void> CSSKeywordValue::setValue(const String& value)
{
    if (value.isEmpty())
        return Exception { ExceptionCode::TypeError };
    
    m_value = value;
    return { };
}

void CSSKeywordValue::serialize(StringBuilder& builder, OptionSet<SerializationArguments>) const
{
    // https://drafts.css-houdini.org/css-typed-om/#keywordvalue-serialization
    serializeIdentifier(m_value, builder);
}

RefPtr<CSSValue> CSSKeywordValue::toCSSValue() const
{
    auto keyword = cssValueKeywordID(m_value);
    if (keyword == CSSValueInvalid)
        return CSSPrimitiveValue::createCustomIdent(m_value);
    return CSSPrimitiveValue::create(keyword);
}

} // namespace WebCore
