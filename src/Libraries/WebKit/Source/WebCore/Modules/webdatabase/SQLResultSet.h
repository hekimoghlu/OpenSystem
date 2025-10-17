/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

#include "ExceptionOr.h"
#include "SQLResultSetRowList.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class SQLResultSet : public ThreadSafeRefCounted<SQLResultSet> {
public:
    static Ref<SQLResultSet> create() { return adoptRef(*new SQLResultSet); }

    SQLResultSetRowList& rows() { return m_rows.get(); }

    ExceptionOr<int64_t> insertId() const;
    int rowsAffected() const { return m_rowsAffected; }

    void setInsertId(int64_t);
    void setRowsAffected(int);

private:
    SQLResultSet();

    Ref<SQLResultSetRowList> m_rows;
    std::optional<int64_t> m_insertId;
    int m_rowsAffected { 0 };
};

inline void SQLResultSet::setInsertId(int64_t id)
{
    ASSERT(!m_insertId);
    m_insertId = id;
}

inline void SQLResultSet::setRowsAffected(int count)
{
    m_rowsAffected = count;
}

} // namespace WebCore
