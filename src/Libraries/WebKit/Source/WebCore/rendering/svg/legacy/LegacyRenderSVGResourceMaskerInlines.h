/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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

#include "LegacyRenderSVGResourceMasker.h"
#include "SVGElementTypeHelpers.h"
#include "SVGMaskElement.h"

namespace WebCore {

inline SVGMaskElement& LegacyRenderSVGResourceMasker::maskElement() const
{
    return downcast<SVGMaskElement>(LegacyRenderSVGResourceContainer::element());
}

inline Ref<SVGMaskElement> LegacyRenderSVGResourceMasker::protectedMaskElement() const
{
    return maskElement();
}

SVGUnitTypes::SVGUnitType LegacyRenderSVGResourceMasker::maskUnits() const
{
    return maskElement().maskUnits();
}

SVGUnitTypes::SVGUnitType LegacyRenderSVGResourceMasker::maskContentUnits() const
{
    return maskElement().maskContentUnits();
}

}
