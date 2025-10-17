/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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

#include "CSSOMColorValue.h"
#include "ExceptionOr.h"

namespace WebCore {

using CSSColorRGBComp = std::variant<double, RefPtr<CSSNumericValue>, String, RefPtr<CSSKeywordValue>>;
using RectifiedCSSColorRGBComp = std::variant<RefPtr<CSSNumericValue>, RefPtr<CSSKeywordValue>>;

class CSSRGB final : public CSSOMColorValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSRGB);
public:
    static ExceptionOr<Ref<CSSRGB>> create(CSSColorRGBComp&&, CSSColorRGBComp&&, CSSColorRGBComp&&, CSSColorPercent&&);

    CSSColorRGBComp r() const;
    ExceptionOr<void> setR(CSSColorRGBComp&&);
    CSSColorRGBComp g() const;
    ExceptionOr<void> setG(CSSColorRGBComp&&);
    CSSColorRGBComp b() const;
    ExceptionOr<void> setB(CSSColorRGBComp&&);
    CSSColorPercent alpha() const;
    ExceptionOr<void> setAlpha(CSSColorPercent&&);

    static ExceptionOr<RectifiedCSSColorRGBComp> rectifyCSSColorRGBComp(CSSColorRGBComp&&);

private:
    CSSRGB(RectifiedCSSColorRGBComp&&, RectifiedCSSColorRGBComp&&, RectifiedCSSColorRGBComp&&, RectifiedCSSColorPercent&&);

    RectifiedCSSColorRGBComp m_red;
    RectifiedCSSColorRGBComp m_green;
    RectifiedCSSColorRGBComp m_blue;
    RectifiedCSSColorPercent m_alpha;
};
    
} // namespace WebCore
