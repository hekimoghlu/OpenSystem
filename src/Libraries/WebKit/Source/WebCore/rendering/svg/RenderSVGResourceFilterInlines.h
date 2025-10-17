/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceFilter.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFilterElement.h"

namespace WebCore {

inline SVGFilterElement& RenderSVGResourceFilter::filterElement() const
{
    return downcast<SVGFilterElement>(RenderSVGResourceContainer::element());
}

inline Ref<SVGFilterElement> RenderSVGResourceFilter::protectedFilterElement() const
{
    return filterElement();
}

inline SVGUnitTypes::SVGUnitType RenderSVGResourceFilter::filterUnits() const
{
    return filterElement().filterUnits();
}

inline SVGUnitTypes::SVGUnitType RenderSVGResourceFilter::primitiveUnits() const
{
    return filterElement().primitiveUnits();
}

} // namespace WebCore
