/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class SQLiteStatement;

class SQLiteStatementAutoResetScope {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SQLiteStatementAutoResetScope, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(SQLiteStatementAutoResetScope);
public:
    WEBCORE_EXPORT explicit SQLiteStatementAutoResetScope(SQLiteStatement* = nullptr);
    WEBCORE_EXPORT SQLiteStatementAutoResetScope(SQLiteStatementAutoResetScope&&);
    WEBCORE_EXPORT SQLiteStatementAutoResetScope& operator=(SQLiteStatementAutoResetScope&&);
    WEBCORE_EXPORT ~SQLiteStatementAutoResetScope();

    explicit operator bool() const { return !!m_statement; }
    bool operator!() const { return !m_statement; }

    SQLiteStatement* get() { return m_statement.get(); }
    SQLiteStatement* operator->() { return m_statement.get(); }

private:
    CheckedPtr<SQLiteStatement> m_statement;
};

} // namespace WebCore
