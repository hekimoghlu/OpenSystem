/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

#include "CSSValue.h"
#include <wtf/FixedVector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSGridLineNamesValue final : public CSSValue {
public:
    static Ref<CSSGridLineNamesValue> create(std::span<const String>);

    std::span<const String> names() const { return m_names; }

    String customCSSText() const;
    bool equals(const CSSGridLineNamesValue& other) const { return m_names == other.m_names; }

private:
    explicit CSSGridLineNamesValue(std::span<const String>);

    FixedVector<String> m_names;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSGridLineNamesValue, isGridLineNamesValue());
