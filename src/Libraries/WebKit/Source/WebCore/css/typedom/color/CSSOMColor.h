/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

class CSSOMColor : public CSSOMColorValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSOMColor);
public:
    template<typename... Args> static Ref<CSSOMColor> create(Args&&... args) { return adoptRef(*new CSSOMColor(std::forward<Args>(args)...)); }

    void setColorSpace(CSSKeywordish);

    const CSSNumberish& alpha() const { return m_alpha; }
    void setAlpha(CSSNumberish alpha) { m_alpha = WTFMove(alpha); }

private:
    CSSOMColor(CSSKeywordish, Vector<CSSColorPercent>, CSSNumberish);

    CSSNumberish m_alpha;
};

} // namespace WebCore
