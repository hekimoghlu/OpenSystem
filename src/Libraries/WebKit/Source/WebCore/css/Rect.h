/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#include "RectBase.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

class Rect final : public RectBase {
public:
    Rect(Ref<CSSPrimitiveValue> top, Ref<CSSPrimitiveValue> right, Ref<CSSPrimitiveValue> bottom, Ref<CSSPrimitiveValue> left)
        : RectBase(WTFMove(top), WTFMove(right), WTFMove(bottom), WTFMove(left))
    { }

    String cssText() const
    {
        return generateCSSString(top().cssText(), right().cssText(), bottom().cssText(), left().cssText());
    }

private:
    static String generateCSSString(const String& top, const String& right, const String& bottom, const String& left)
    {
        return makeString("rect("_s, top, ", "_s, right, ", "_s, bottom, ", "_s, left, ')');
    }
};

} // namespace WebCore
