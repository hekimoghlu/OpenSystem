/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "CalculationValue.h"

#include "CalculationTree+Copy.h"
#include "CalculationTree+Evaluation.h"
#include <cmath>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<CalculationValue> CalculationValue::create(Calculation::Category category, Calculation::Range range, Calculation::Tree&& tree)
{
    return adoptRef(*new CalculationValue(category, range, WTFMove(tree)));
}

CalculationValue::CalculationValue(Calculation::Category category, Calculation::Range range, Calculation::Tree&& tree)
    : m_category(category)
    , m_range(range)
    , m_tree(WTFMove(tree))
{
}

CalculationValue::~CalculationValue() = default;

double CalculationValue::evaluate(double percentResolutionLength) const
{
    auto result = Calculation::evaluate(m_tree, percentResolutionLength);
    if (std::isnan(result))
        return 0;
    return std::clamp(result, m_range.min, m_range.max);
}

Calculation::Tree CalculationValue::copyTree() const
{
    return Calculation::copy(m_tree);
}

Calculation::Child CalculationValue::copyRoot() const
{
    auto tree = copyTree();
    return { WTFMove(tree.root) };
}

bool CalculationValue::operator==(const CalculationValue& other) const
{
    return m_tree == other.m_tree;
}

TextStream& operator<<(TextStream& ts, const CalculationValue& value)
{
    return ts << "calc(" << value.tree() << ")";
}

} // namespace WebCore
