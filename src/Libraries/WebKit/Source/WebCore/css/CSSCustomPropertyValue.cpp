/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include "CSSCustomPropertyValue.h"

#include "CSSCalcValue.h"
#include "CSSFunctionValue.h"
#include "CSSMarkup.h"
#include "CSSParserIdioms.h"
#include "CSSTokenizer.h"
#include "ColorSerialization.h"
#include "ComputedStyleExtractor.h"
#include "RenderStyle.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

Ref<CSSCustomPropertyValue> CSSCustomPropertyValue::createEmpty(const AtomString& name)
{
    static NeverDestroyed<Ref<CSSVariableData>> empty { CSSVariableData::create({ }) };
    return createSyntaxAll(name, Ref { empty.get() });
}

Ref<CSSCustomPropertyValue> CSSCustomPropertyValue::createWithID(const AtomString& name, CSSValueID id)
{
    ASSERT(WebCore::isCSSWideKeyword(id) || id == CSSValueInvalid);
    return adoptRef(*new CSSCustomPropertyValue(name, { id }));
}

bool CSSCustomPropertyValue::equals(const CSSCustomPropertyValue& other) const
{
    if (m_name != other.m_name || m_value.index() != other.m_value.index())
        return false;
    return WTF::switchOn(m_value, [&](const Ref<CSSVariableReferenceValue>& value) {
        auto& otherValue = std::get<Ref<CSSVariableReferenceValue>>(other.m_value);
        return value.ptr() == otherValue.ptr() || value.get() == otherValue.get();
    }, [&](const CSSValueID& value) {
        return value == std::get<CSSValueID>(other.m_value);
    }, [&](const Ref<CSSVariableData>& value) {
        auto& otherValue = std::get<Ref<CSSVariableData>>(other.m_value);
        return value.ptr() == otherValue.ptr() || value.get() == otherValue.get();
    }, [&](const SyntaxValue& value) {
        return value == std::get<SyntaxValue>(other.m_value);
    }, [&](const SyntaxValueList& value) {
        return value == std::get<SyntaxValueList>(other.m_value);
    });
}

String CSSCustomPropertyValue::customCSSText() const
{
    auto serializeSyntaxValue = [](const SyntaxValue& syntaxValue) -> String {
        return WTF::switchOn(syntaxValue, [&](const Length& value) {
            if (value.type() == LengthType::Calculated) {
                // FIXME: Implement serialization for CalculationValue directly.
                auto calcValue = CSSCalcValue::create(value.calculationValue(), RenderStyle::defaultStyle());
                return calcValue->cssText();
            }
            return CSSPrimitiveValue::create(value, RenderStyle::defaultStyle())->cssText();
        }, [&](const NumericSyntaxValue& value) {
            return CSSPrimitiveValue::create(value.value, value.unitType)->cssText();
        }, [&](const Style::Color& value) {
            return serializationForCSS(value);
        }, [&](const RefPtr<StyleImage>& value) {
            // FIXME: This is not right for gradients that use `currentcolor`. There should be a way preserve it.
            return value->computedStyleValue(RenderStyle::defaultStyle())->cssText();
        }, [&](const URL& value) {
            return serializeURL(value.string());
        }, [&](const String& value) {
            return value;
        }, [&](const TransformSyntaxValue& value) {
            auto cssValue = transformOperationAsCSSValue(value.transform, RenderStyle::defaultStyle());
            if (!cssValue)
                return emptyString();
            return cssValue->cssText();
        });
    };

    auto serialize = [&] {
        return WTF::switchOn(m_value, [&](const Ref<CSSVariableReferenceValue>& value) {
            return value->cssText();
        }, [&](const CSSValueID& value) {
            return nameString(value).string();
        }, [&](const Ref<CSSVariableData>& value) {
            return value->serialize();
        }, [&](const SyntaxValue& syntaxValue) {
            return serializeSyntaxValue(syntaxValue);
        }, [&](const SyntaxValueList& syntaxValueList) {
            StringBuilder builder;
            auto separator = separatorCSSText(syntaxValueList.separator);
            for (auto& syntaxValue : syntaxValueList.values) {
                if (!builder.isEmpty())
                    builder.append(separator);
                builder.append(serializeSyntaxValue(syntaxValue));
            }
            return builder.toString();
        });
    };

    if (m_cachedCSSText.isNull())
        m_cachedCSSText = serialize();

    return m_cachedCSSText;
}

const Vector<CSSParserToken>& CSSCustomPropertyValue::tokens() const
{
    static NeverDestroyed<Vector<CSSParserToken>> emptyTokens;

    return WTF::switchOn(m_value, [&](const Ref<CSSVariableReferenceValue>&) -> const Vector<CSSParserToken>& {
        ASSERT_NOT_REACHED();
        return emptyTokens;
    }, [&](const CSSValueID&) -> const Vector<CSSParserToken>& {
        // Do nothing.
        return emptyTokens;
    }, [&](const Ref<CSSVariableData>& value) -> const Vector<CSSParserToken>& {
        return value->tokens();
    }, [&](auto&) -> const Vector<CSSParserToken>& {
        if (!m_cachedTokens) {
            CSSTokenizer tokenizer { customCSSText() };
            m_cachedTokens = CSSVariableData::create(tokenizer.tokenRange());
        }
        return m_cachedTokens->tokens();
    });
}

bool CSSCustomPropertyValue::containsCSSWideKeyword() const
{
    return std::holds_alternative<CSSValueID>(m_value) && WebCore::isCSSWideKeyword(std::get<CSSValueID>(m_value));
}

Ref<const CSSVariableData> CSSCustomPropertyValue::asVariableData() const
{
    return WTF::switchOn(m_value, [&](const Ref<CSSVariableData>& value) -> Ref<const CSSVariableData> {
        return value.get();
    }, [&](const Ref<CSSVariableReferenceValue>& value) -> Ref<const CSSVariableData> {
        return value->data();
    }, [&](auto&) -> Ref<const CSSVariableData> {
        return CSSVariableData::create(tokens());
    });
}

bool CSSCustomPropertyValue::isCurrentColor() const
{
    // FIXME: Registered properties?
    auto tokenRange = switchOn(m_value, [&](const Ref<CSSVariableReferenceValue>& variableReferenceValue) {
        return variableReferenceValue->data().tokenRange();
    }, [&](const Ref<CSSVariableData>& data) {
        return data->tokenRange();
    }, [&](auto&) {
        return CSSParserTokenRange { };
    });

    if (tokenRange.atEnd())
        return false;

    auto token = tokenRange.consumeIncludingWhitespace();
    if (!tokenRange.atEnd())
        return false;

    // FIXME: This should probably check all tokens.
    return token.id() == CSSValueCurrentcolor;
}

bool CSSCustomPropertyValue::isAnimatable() const
{
    return std::holds_alternative<SyntaxValue>(m_value) || std::holds_alternative<SyntaxValueList>(m_value);
}

static bool mayDependOnBaseURL(const CSSCustomPropertyValue::SyntaxValue& syntaxValue)
{
    return WTF::switchOn(syntaxValue,
        [](const Length&) {
            return false;
        },
        [](const CSSCustomPropertyValue::NumericSyntaxValue&) {
            return false;
        },
        [](const Style::Color&) {
            return false;
        },
        [](const RefPtr<StyleImage>&) {
            return true;
        },
        [](const URL&) {
            return true;
        },
        [](const String&) {
            return false;
        },
        [](const CSSCustomPropertyValue::TransformSyntaxValue&) {
            return false;
        });
}

bool CSSCustomPropertyValue::customMayDependOnBaseURL() const
{
    return WTF::switchOn(m_value,
        [](const Ref<CSSVariableReferenceValue>&) {
            return false;
        },
        [](const CSSValueID&) {
            return false;
        },
        [](const Ref<CSSVariableData>&) {
            return false;
        },
        [](const SyntaxValue& syntaxValue) {
            return WebCore::mayDependOnBaseURL(syntaxValue);
        },
        [](const SyntaxValueList& syntaxValueList) {
            for (auto& syntaxValue : syntaxValueList.values) {
                if (WebCore::mayDependOnBaseURL(syntaxValue))
                    return true;
            }
            return false;
        });
}

}
