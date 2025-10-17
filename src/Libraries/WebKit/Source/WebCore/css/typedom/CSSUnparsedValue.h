/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#include <variant>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename> class ExceptionOr;
class CSSOMVariableReferenceValue;
class CSSParserTokenRange;
using CSSUnparsedSegment = std::variant<String, RefPtr<CSSOMVariableReferenceValue>>;

class CSSUnparsedValue final : public CSSStyleValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSUnparsedValue);
public:
    static Ref<CSSUnparsedValue> create(Vector<CSSUnparsedSegment>&&);
    static Ref<CSSUnparsedValue> create(CSSParserTokenRange);

    virtual ~CSSUnparsedValue();

    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;
    size_t length() const { return m_segments.size(); }
    
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_segments.size(); }
    std::optional<CSSUnparsedSegment> item(size_t);
    ExceptionOr<CSSUnparsedSegment> setItem(size_t, CSSUnparsedSegment&&);

    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSUnparsedValue; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    explicit CSSUnparsedValue(Vector<CSSUnparsedSegment>&& segments);
    
    Vector<CSSUnparsedSegment> m_segments;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSUnparsedValue)
    static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSUnparsedValue; }
SPECIALIZE_TYPE_TRAITS_END()
