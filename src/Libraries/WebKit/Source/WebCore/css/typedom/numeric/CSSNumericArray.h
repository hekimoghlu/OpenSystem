/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

#include "CSSNumericValue.h"

namespace WebCore {

class CSSNumericArray : public RefCounted<CSSNumericArray> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSNumericArray);
public:
    static Ref<CSSNumericArray> create(FixedVector<CSSNumberish>&&);
    static Ref<CSSNumericArray> create(Vector<Ref<CSSNumericValue>>&&);
    size_t length() const { return m_array.size(); };
    RefPtr<CSSNumericValue> item(size_t index);
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_array.size(); }
    const Vector<Ref<CSSNumericValue>>& array() const { return m_array; }
    void forEach(Function<void(const CSSNumericValue&, bool first)>);

private:
    Vector<Ref<CSSNumericValue>> m_array;
    CSSNumericArray(Vector<Ref<CSSNumericValue>>&&);
};

} // namespace WebCore
