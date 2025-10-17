/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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

#include "CSSKeywordValue.h"
#include "CSSNumericValue.h"
#include "CSSStyleValue.h"

namespace WebCore {

class CSSKeywordValue;

using CSSKeywordish = std::variant<String, RefPtr<CSSKeywordValue>>;
using CSSColorPercent = std::variant<double, RefPtr<CSSNumericValue>, String, RefPtr<CSSKeywordValue>>;
using RectifiedCSSColorPercent = std::variant<RefPtr<CSSNumericValue>, RefPtr<CSSKeywordValue>>;
using CSSColorNumber = std::variant<double, RefPtr<CSSNumericValue>, String, RefPtr<CSSKeywordValue>>;
using RectifiedCSSColorNumber = std::variant<RefPtr<CSSNumericValue>, RefPtr<CSSKeywordValue>>;
using CSSColorAngle = std::variant<double, RefPtr<CSSNumericValue>, String, RefPtr<CSSKeywordValue>>;
using RectifiedCSSColorAngle = std::variant<RefPtr<CSSNumericValue>, RefPtr<CSSKeywordValue>>;

class CSSOMColorValue : public CSSStyleValue {
public:
    RefPtr<CSSKeywordValue> colorSpace();
    RefPtr<CSSOMColorValue> to(CSSKeywordish);
    static std::variant<RefPtr<CSSOMColorValue>, RefPtr<CSSStyleValue>> parse(const String&);

    static ExceptionOr<RectifiedCSSColorPercent> rectifyCSSColorPercent(CSSColorPercent&&);
    static ExceptionOr<RectifiedCSSColorAngle> rectifyCSSColorAngle(CSSColorAngle&&);
    static ExceptionOr<RectifiedCSSColorNumber> rectifyCSSColorNumber(CSSColorNumber&&);
    static CSSColorPercent toCSSColorPercent(const RectifiedCSSColorPercent&);
    static CSSColorPercent toCSSColorPercent(const CSSNumberish&);
    static CSSColorAngle toCSSColorAngle(const RectifiedCSSColorAngle&);
    static CSSColorNumber toCSSColorNumber(const RectifiedCSSColorNumber&);

    RefPtr<CSSValue> toCSSValue() const final;
};

} // namespace WebCore
