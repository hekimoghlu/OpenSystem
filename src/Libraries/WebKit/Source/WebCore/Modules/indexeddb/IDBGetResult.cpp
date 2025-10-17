/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "IDBGetResult.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBGetResult);

IDBGetResult::IDBGetResult(const IDBGetResult& that, IsolatedCopyTag)
{
    isolatedCopy(that, *this);
}

IDBGetResult IDBGetResult::isolatedCopy() const
{
    return { *this, IsolatedCopy };
}

void IDBGetResult::isolatedCopy(const IDBGetResult& source, IDBGetResult& destination)
{
    destination.m_value = source.m_value.isolatedCopy();
    destination.m_keyData = source.m_keyData.isolatedCopy();
    destination.m_primaryKeyData = source.m_primaryKeyData.isolatedCopy();
    destination.m_keyPath = crossThreadCopy(source.m_keyPath);
    destination.m_isDefined = source.m_isDefined;
    destination.m_prefetchedRecords = crossThreadCopy(source.m_prefetchedRecords);
}

void IDBGetResult::setValue(IDBValue&& value)
{
    m_value = WTFMove(value);
}

} // namespace WebCore
