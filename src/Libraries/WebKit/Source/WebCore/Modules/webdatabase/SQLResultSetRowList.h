/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include "SQLValue.h"

namespace WebCore {

class SQLResultSetRowList : public RefCounted<SQLResultSetRowList> {
public:
    static Ref<SQLResultSetRowList> create() { return adoptRef(*new SQLResultSetRowList); }

    const Vector<String>& columnNames() const { return m_columns; }
    const Vector<SQLValue>& values() const { return m_result; }

    void addColumn(const String& name) { m_columns.append(name); }
    void addResult(const SQLValue& result) { m_result.append(result); }

    unsigned length() const;
    ExceptionOr<Vector<KeyValuePair<String, SQLValue>>> item(unsigned index) const;

private:
    SQLResultSetRowList() { }

    Vector<String> m_columns;
    Vector<SQLValue> m_result;
};

} // namespace WebCore
