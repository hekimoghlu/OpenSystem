/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "CSSNumericArray.h"

#include "ExceptionOr.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSNumericArray);

Ref<CSSNumericArray> CSSNumericArray::create(FixedVector<CSSNumberish>&& numberishes)
{
    return adoptRef(*new CSSNumericArray(WTF::map(WTFMove(numberishes), CSSNumericValue::rectifyNumberish)));
}

Ref<CSSNumericArray> CSSNumericArray::create(Vector<Ref<CSSNumericValue>>&& values)
{
    return adoptRef(*new CSSNumericArray(WTFMove(values)));
}

CSSNumericArray::CSSNumericArray(Vector<Ref<CSSNumericValue>>&& values)
    : m_array(WTFMove(values))
{
}

RefPtr<CSSNumericValue> CSSNumericArray::item(size_t index)
{
    if (index >= m_array.size())
        return nullptr;
    return m_array[index].copyRef();
}

void CSSNumericArray::forEach(Function<void(const CSSNumericValue&, bool first)> function)
{
    for (size_t i = 0; i < m_array.size(); ++i)
        function(m_array[i], !i);
}

} // namespace WebCore
