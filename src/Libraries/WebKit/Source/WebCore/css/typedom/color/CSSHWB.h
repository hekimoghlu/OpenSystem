/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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

class CSSHWB final : public CSSOMColorValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSHWB);
public:
    static ExceptionOr<Ref<CSSHWB>> create(Ref<CSSNumericValue>&& hue, CSSNumberish&& whiteness, CSSNumberish&& blackness, CSSNumberish&& alpha);

    CSSNumericValue& h() const;
    ExceptionOr<void> setH(Ref<CSSNumericValue>&&);
    CSSNumberish w() const;
    ExceptionOr<void> setW(CSSNumberish&&);
    CSSNumberish b() const;
    ExceptionOr<void> setB(CSSNumberish&&);
    CSSNumberish alpha() const;
    ExceptionOr<void> setAlpha(CSSNumberish&&);

private:
    CSSHWB(Ref<CSSNumericValue>&& hue, Ref<CSSNumericValue>&& whiteness, Ref<CSSNumericValue>&& m_blackness, Ref<CSSNumericValue>&& alpha);

    Ref<CSSNumericValue> m_hue;
    Ref<CSSNumericValue> m_whiteness;
    Ref<CSSNumericValue> m_blackness;
    Ref<CSSNumericValue> m_alpha;
};
    
} // namespace WebCore
