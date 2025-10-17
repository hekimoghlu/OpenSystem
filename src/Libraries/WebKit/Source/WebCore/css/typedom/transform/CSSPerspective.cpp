/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include "CSSPerspective.h"

#include "CSSFunctionValue.h"
#include "CSSKeywordValue.h"
#include "CSSNumericFactory.h"
#include "CSSNumericValue.h"
#include "CSSStyleValueFactory.h"
#include "CSSUnitValue.h"
#include "DOMMatrix.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSPerspective);

static ExceptionOr<CSSPerspectiveValue> checkLength(CSSPerspectiveValue length)
{
    // https://drafts.css-houdini.org/css-typed-om/#dom-cssperspective-cssperspective
    auto checkKeywordValue = [] (RefPtr<CSSKeywordValue> value) -> ExceptionOr<CSSPerspectiveValue> {
        RELEASE_ASSERT(value);
        if (!equalLettersIgnoringASCIICase(value->value(), "none"_s))
            return Exception { ExceptionCode::TypeError };
        return { WTFMove(value) };
    };
    return WTF::switchOn(WTFMove(length),
        [] (RefPtr<CSSNumericValue> value) -> ExceptionOr<CSSPerspectiveValue> {
            if (value && !value->type().matches<CSSNumericBaseType::Length>())
                return Exception { ExceptionCode::TypeError };
            return { WTFMove(value) };
        }, [&] (String value) {
            return checkKeywordValue(CSSKeywordValue::rectifyKeywordish(WTFMove(value)));
        }, checkKeywordValue);
}

ExceptionOr<Ref<CSSPerspective>> CSSPerspective::create(CSSPerspectiveValue length)
{
    auto checkedLength = checkLength(WTFMove(length));
    if (checkedLength.hasException())
        return checkedLength.releaseException();
    return adoptRef(*new CSSPerspective(checkedLength.releaseReturnValue()));
}

ExceptionOr<Ref<CSSPerspective>> CSSPerspective::create(Ref<const CSSFunctionValue> cssFunctionValue)
{
    if (cssFunctionValue->name() != CSSValuePerspective) {
        ASSERT_NOT_REACHED();
        return CSSPerspective::create("none"_s);
    }

    if (cssFunctionValue->size() != 1 || !cssFunctionValue->item(0)) {
        ASSERT_NOT_REACHED();
        return Exception { ExceptionCode::TypeError, "Unexpected number of values."_s };
    }

    auto keywordOrNumeric = CSSStyleValueFactory::reifyValue(*cssFunctionValue->item(0), std::nullopt);
    if (keywordOrNumeric.hasException())
        return keywordOrNumeric.releaseException();
    auto& keywordOrNumericValue = keywordOrNumeric.returnValue().get();
    return [&]() -> ExceptionOr<Ref<CSSPerspective>> {
        if (auto* keywordValue = dynamicDowncast<CSSKeywordValue>(keywordOrNumericValue))
            return CSSPerspective::create(keywordValue);
        if (auto* numericValue = dynamicDowncast<CSSNumericValue>(keywordOrNumericValue))
            return CSSPerspective::create(numericValue);
        return Exception { ExceptionCode::TypeError, "Expected a CSSNumericValue."_s };
    }();
}

CSSPerspective::CSSPerspective(CSSPerspectiveValue length)
    : CSSTransformComponent(Is2D::No)
    , m_length(WTFMove(length))
{
}

CSSPerspective::~CSSPerspective() = default;

ExceptionOr<void> CSSPerspective::setLength(CSSPerspectiveValue length)
{
    auto checkedLength = checkLength(WTFMove(length));
    if (checkedLength.hasException())
        return checkedLength.releaseException();
    m_length = checkedLength.releaseReturnValue();
    return { };
}

void CSSPerspective::setIs2D(bool)
{
    // https://drafts.css-houdini.org/css-typed-om/#dom-cssperspective-is2d says to do nothing here.
}

void CSSPerspective::serialize(StringBuilder& builder) const
{
    // https://drafts.css-houdini.org/css-typed-om/#serialize-a-cssperspective
    builder.append("perspective("_s);
    WTF::switchOn(m_length,
        [&] (const RefPtr<CSSNumericValue>& value) {
            if (auto* unitValue = dynamicDowncast<CSSUnitValue>(value.get()); unitValue && unitValue->value() < 0.0) {
                builder.append("calc("_s);
                value->serialize(builder);
                builder.append(')');
                return;
            }
            if (value)
                value->serialize(builder);
        }, [&] (const String& value) {
            builder.append(value);
        }, [&] (const RefPtr<CSSKeywordValue>& value) {
            if (CSSStyleValue* styleValue = value.get())
                styleValue->serialize(builder);
        });
    builder.append(')');
}

ExceptionOr<Ref<DOMMatrix>> CSSPerspective::toMatrix()
{
    if (!std::holds_alternative<RefPtr<CSSNumericValue>>(m_length))
        return { DOMMatrix::create({ }, DOMMatrixReadOnly::Is2D::Yes) };

    RefPtr length = dynamicDowncast<CSSUnitValue>(std::get<RefPtr<CSSNumericValue>>(m_length));
    if (!length)
        return Exception { ExceptionCode::TypeError };

    auto valuePx = length->convertTo(CSSUnitType::CSS_PX);
    if (!valuePx)
        return Exception { ExceptionCode::TypeError, "Length unit is not compatible with 'px'"_s };

    TransformationMatrix matrix { };
    matrix.applyPerspective(valuePx->value());

    return { DOMMatrix::create(WTFMove(matrix), DOMMatrixReadOnly::Is2D::No) };
}

RefPtr<CSSValue> CSSPerspective::toCSSValue() const
{
    RefPtr<CSSValue> length;
    switchOn(m_length, [&](const RefPtr<CSSNumericValue>& numericValue) {
        length = numericValue->toCSSValue();
    }, [&](const String&) {
        // FIXME: Implement this.
    }, [&](const RefPtr<CSSKeywordValue>& keywordValue) {
        length = keywordValue->toCSSValue();
    });
    if (!length)
        return nullptr;

    return CSSFunctionValue::create(CSSValuePerspective, length.releaseNonNull());
}

} // namespace WebCore
