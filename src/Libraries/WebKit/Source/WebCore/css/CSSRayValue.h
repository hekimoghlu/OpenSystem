/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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

#include "CSSRayFunction.h"
#include "RenderStyleConstants.h"

namespace WebCore {

// Class containing the value of a ray() function, as used in offset-path:
// https://drafts.fxtf.org/motion-1/#funcdef-offset-path-ray.
class CSSRayValue final : public CSSValue {
public:
    static Ref<CSSRayValue> create(CSS::RayFunction ray, CSSBoxType coordinateBox = CSSBoxType::BoxMissing)
    {
        return adoptRef(*new CSSRayValue(WTFMove(ray), coordinateBox));
    }


    const CSS::RayFunction& ray() const { return m_ray; }
    CSSBoxType coordinateBox() const { return m_coordinateBox; }

    String customCSSText() const;
    bool equals(const CSSRayValue&) const;
    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

private:
    CSSRayValue(CSS::RayFunction ray, CSSBoxType coordinateBox)
        : CSSValue(ClassType::Ray)
        , m_ray(WTFMove(ray))
        , m_coordinateBox(coordinateBox)
    {
    }

    CSS::RayFunction m_ray;
    CSSBoxType m_coordinateBox;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSRayValue, isRayValue())
