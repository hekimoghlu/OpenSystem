/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#include "CSSUnparsedValue.h"

#include "CSSOMVariableReferenceValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSTokenizer.h"
#include "CSSVariableReferenceValue.h"
#include "ExceptionOr.h"
#include <variant>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringView.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSUnparsedValue);

Ref<CSSUnparsedValue> CSSUnparsedValue::create(Vector<CSSUnparsedSegment>&& segments)
{
    return adoptRef(*new CSSUnparsedValue(WTFMove(segments)));
}

Ref<CSSUnparsedValue> CSSUnparsedValue::create(CSSParserTokenRange tokens)
{
    // This function assumes that tokens have the correct syntax. Otherwise asserts would be triggered.
    StringBuilder builder;
    Vector<Vector<CSSUnparsedSegment>> segmentStack;
    segmentStack.append({ });
    
    Vector<std::optional<StringView>> identifiers;
    
    while (!tokens.atEnd()) {
        auto currentToken = tokens.consume();
        
        if (currentToken.type() == FunctionToken || currentToken.type() == LeftParenthesisToken) {
            if (currentToken.functionId() == CSSValueVar) {
                if (!builder.isEmpty()) {
                    segmentStack.last().append(builder.toString());
                    builder.clear();
                }
                tokens.consumeWhitespace();
                auto identToken = tokens.consumeIncludingWhitespace();
                // Token after whitespace consumption must be variable reference identifier
                ASSERT(identToken.type() == IdentToken);
                if (tokens.peek().type() == CommaToken) {
                    // Fallback present
                    identifiers.append(StringView(identToken.value()));
                    segmentStack.append({ });
                    tokens.consume();
                } else if (tokens.peek().type() == RightParenthesisToken) {
                    // No fallback
                    auto variableReference = CSSOMVariableReferenceValue::create(identToken.value().toString());
                    ASSERT(!variableReference.hasException());
                    segmentStack.last().append(CSSUnparsedSegment { RefPtr<CSSOMVariableReferenceValue> { variableReference.releaseReturnValue() } });
                    tokens.consume();
                } else
                    ASSERT_NOT_REACHED();
                
            } else {
                currentToken.serialize(builder);
                identifiers.append(std::nullopt);
            }
        } else if (currentToken.type() == RightParenthesisToken) {
            ASSERT(segmentStack.size());
            if (!builder.isEmpty())
                segmentStack.last().append(builder.toString());
            builder.clear();
            ASSERT(!identifiers.isEmpty());
            
            if (auto topIdentifier = identifiers.takeLast()) {
                auto variableReference = CSSOMVariableReferenceValue::create(topIdentifier->toString(), CSSUnparsedValue::create(segmentStack.takeLast()));
                ASSERT(!variableReference.hasException());
                segmentStack.last().append(variableReference.releaseReturnValue());
            } else
                currentToken.serialize(builder);
        } else
            currentToken.serialize(builder);
    }
    ASSERT(segmentStack.size() == 1);
    if (!builder.isEmpty())
        segmentStack.last().append(builder.toString());
    
    return CSSUnparsedValue::create(WTFMove(segmentStack.last()));
}

CSSUnparsedValue::CSSUnparsedValue(Vector<CSSUnparsedSegment>&& segments)
    : m_segments(WTFMove(segments))
{
}

CSSUnparsedValue::~CSSUnparsedValue() = default;

void CSSUnparsedValue::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    for (auto& segment : m_segments) {
        std::visit(WTF::makeVisitor([&] (const String& value) {
            builder.append(value);
        }, [&] (const RefPtr<CSSOMVariableReferenceValue>& value) {
            value->serialize(builder, arguments);
        }), segment);
    }
}

std::optional<CSSUnparsedSegment> CSSUnparsedValue::item(size_t index)
{
    if (index >= m_segments.size())
        return std::nullopt;
    return CSSUnparsedSegment { m_segments[index] };
}

ExceptionOr<CSSUnparsedSegment> CSSUnparsedValue::setItem(size_t index, CSSUnparsedSegment&& val)
{
    if (index > m_segments.size())
        return Exception { ExceptionCode::RangeError, makeString("Index "_s, index, " exceeds index range for unparsed segments."_s) };
    if (index == m_segments.size())
        m_segments.append(WTFMove(val));
    else
        m_segments[index] = WTFMove(val);
    return CSSUnparsedSegment { m_segments[index] };
}

RefPtr<CSSValue> CSSUnparsedValue::toCSSValue() const
{
    CSSTokenizer tokenizer(toString());
    return CSSVariableReferenceValue::create(tokenizer.tokenRange(), strictCSSParserContext());
}

} // namespace WebCore
