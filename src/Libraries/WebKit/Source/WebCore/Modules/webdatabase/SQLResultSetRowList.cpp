/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#include "SQLResultSetRowList.h"

namespace WebCore {

unsigned SQLResultSetRowList::length() const
{
    if (m_result.isEmpty())
        return 0;

    ASSERT(m_result.size() % m_columns.size() == 0);

    return m_result.size() / m_columns.size();
}

ExceptionOr<Vector<KeyValuePair<String, SQLValue>>> SQLResultSetRowList::item(unsigned index) const
{
    if (index >= length())
        return Exception { ExceptionCode::IndexSizeError };

    Vector<KeyValuePair<String, SQLValue>> result;

    unsigned numberOfColumns = m_columns.size();
    unsigned valuesIndex = index * numberOfColumns;
    for (unsigned i = 0; i < numberOfColumns; i++)
        result.append({ m_columns[i], m_result[valuesIndex + i] });

    return result;
}

}
