/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

#include "CSSPrimitiveValue.h"
#include "CSSValue.h"

namespace WebCore {

class CSSGridLineValue final : public CSSValue {
public:
    static Ref<CSSGridLineValue> create(RefPtr<CSSPrimitiveValue>&&, RefPtr<CSSPrimitiveValue>&&, RefPtr<CSSPrimitiveValue>&&);

    String customCSSText() const;
    bool equals(const CSSGridLineValue& other) const;

    CSSPrimitiveValue* spanValue() const { return m_spanValue.get(); }
    CSSPrimitiveValue* numericValue() const { return m_numericValue.get(); }
    CSSPrimitiveValue* gridLineName() const { return m_gridLineName.get(); }

private:
    explicit CSSGridLineValue(RefPtr<CSSPrimitiveValue>&&, RefPtr<CSSPrimitiveValue>&&, RefPtr<CSSPrimitiveValue>&&);
    RefPtr<CSSPrimitiveValue> m_spanValue;
    RefPtr<CSSPrimitiveValue> m_numericValue;
    RefPtr<CSSPrimitiveValue> m_gridLineName;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSGridLineValue, isGridLineValue());
