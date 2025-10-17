/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "CSSMathValue.h"

#include "CSSCalcTree.h"
#include "CSSCalcValue.h"
#include "CSSPrimitiveValue.h"
#include "Length.h"

namespace WebCore {

RefPtr<CSSValue> CSSMathValue::toCSSValue() const
{
    auto node = toCalcTreeNode();
    if (!node)
        return nullptr;

    auto type = CSSCalc::getType(*node);
    auto category = type.calculationCategory();
    if (!category)
        return nullptr;

    return CSSPrimitiveValue::create(CSSCalcValue::create(*category, CSS::All, CSSCalc::Tree {
        .root = WTFMove(*node),
        .type = type,
        .stage = CSSCalc::Stage::Specified,
    }));
}

} // namespace WebCore
