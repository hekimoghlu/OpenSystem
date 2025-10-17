/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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

#include "CSSStyleValue.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename> class ExceptionOr;
class CSSKeywordValue;
using CSSKeywordish = std::variant<String, RefPtr<CSSKeywordValue>>;

class CSSKeywordValue final : public CSSStyleValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSKeywordValue);
public:
    static ExceptionOr<Ref<CSSKeywordValue>> create(const String&);
    
    const String& value() const { return m_value; }
    ExceptionOr<void> setValue(const String&);
    
    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSKeywordValue; }
    
    static Ref<CSSKeywordValue> rectifyKeywordish(CSSKeywordish&&);

    RefPtr<CSSValue> toCSSValue() const final;

private:
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;

    explicit CSSKeywordValue(const String& value)
        : m_value(value) { }
    String m_value;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSKeywordValue)
    static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSKeywordValue; }
SPECIALIZE_TYPE_TRAITS_END()
