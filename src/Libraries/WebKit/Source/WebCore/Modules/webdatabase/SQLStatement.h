/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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

#include "SQLCallbackWrapper.h"
#include "SQLStatementCallback.h"
#include "SQLStatementErrorCallback.h"
#include "SQLValue.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Database;
class SQLError;
class SQLResultSet;
class SQLTransactionBackend;

class SQLStatement final {
    WTF_MAKE_TZONE_ALLOCATED(SQLStatement);
public:
    SQLStatement(Database&, const String&, Vector<SQLValue>&&, RefPtr<SQLStatementCallback>&&, RefPtr<SQLStatementErrorCallback>&&, int permissions);
    ~SQLStatement();

    bool execute(Database&);
    bool lastExecutionFailedDueToQuota() const;

    bool hasStatementCallback() const { return m_statementCallbackWrapper.hasCallback(); }
    bool hasStatementErrorCallback() const { return m_statementErrorCallbackWrapper.hasCallback(); }
    bool performCallback(SQLTransaction&);

    void setDatabaseDeletedError();
    void setVersionMismatchedError();

    SQLError* sqlError() const;
    SQLResultSet* sqlResultSet() const;

private:
    void setFailureDueToQuota();
    void clearFailureDueToQuota();

    String m_statement;
    Vector<SQLValue> m_arguments;
    SQLCallbackWrapper<SQLStatementCallback> m_statementCallbackWrapper;
    SQLCallbackWrapper<SQLStatementErrorCallback> m_statementErrorCallbackWrapper;

    RefPtr<SQLError> m_error;
    RefPtr<SQLResultSet> m_resultSet;

    int m_permissions;
};

} // namespace WebCore
