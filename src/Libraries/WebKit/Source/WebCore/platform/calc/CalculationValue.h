/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

#include "CalculationRange.h"
#include "CalculationTree.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

class CalculationValue : public RefCounted<CalculationValue> {
    WTF_MAKE_FAST_COMPACT_ALLOCATED;
public:
    WEBCORE_EXPORT static Ref<CalculationValue> create(Calculation::Category, Calculation::Range, Calculation::Tree&&);
    WEBCORE_EXPORT ~CalculationValue();

    double evaluate(double percentResolutionLength) const;

    Calculation::Category category() const { return m_category; }
    Calculation::Range range() const { return m_range; }

    const Calculation::Tree& tree() const { return m_tree; }
    Calculation::Tree copyTree() const;
    Calculation::Child copyRoot() const;

    WEBCORE_EXPORT bool operator==(const CalculationValue&) const;

private:
    CalculationValue(Calculation::Category, Calculation::Range, Calculation::Tree&&);

    Calculation::Category m_category;
    Calculation::Range m_range;
    Calculation::Tree m_tree;
};

TextStream& operator<<(TextStream&, const CalculationValue&);

} // namespace WebCore

