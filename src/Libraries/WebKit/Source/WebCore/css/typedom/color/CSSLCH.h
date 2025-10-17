/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

namespace WebCore {

class CSSLCH final : public CSSOMColorValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSLCH);
public:
    static ExceptionOr<Ref<CSSLCH>> create(CSSColorPercent&& lightness, CSSColorPercent&& chroma, CSSColorAngle&& hue, CSSColorPercent&& alpha);

    CSSColorPercent l() const;
    ExceptionOr<void> setL(CSSColorPercent&&);
    CSSColorPercent c() const;
    ExceptionOr<void> setC(CSSColorPercent&&);
    CSSColorAngle h() const;
    ExceptionOr<void> setH(CSSColorAngle&&);
    CSSColorPercent alpha() const;
    ExceptionOr<void> setAlpha(CSSColorPercent&&);

private:
    CSSLCH(RectifiedCSSColorPercent&& lightness, RectifiedCSSColorPercent&& chroma, RectifiedCSSColorAngle&& hue, RectifiedCSSColorPercent&& alpha);

    RectifiedCSSColorPercent m_lightness;
    RectifiedCSSColorPercent m_chroma;
    RectifiedCSSColorAngle m_hue;
    RectifiedCSSColorPercent m_alpha;
};
    
} // namespace WebCore
