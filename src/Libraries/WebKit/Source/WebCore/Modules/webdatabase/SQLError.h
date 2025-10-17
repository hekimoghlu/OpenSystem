/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

class SQLError : public ThreadSafeRefCounted<SQLError> {
public:
    static Ref<SQLError> create(unsigned code, String&& message) { return adoptRef(*new SQLError(code, WTFMove(message))); }
    static Ref<SQLError> create(unsigned code, ASCIILiteral message, int sqliteCode)
    {
        return create(code, makeString(message, " ("_s, sqliteCode, ')'));
    }
    static Ref<SQLError> create(unsigned code, ASCIILiteral message, int sqliteCode, const char* sqliteMessage)
    {
        return create(code, makeString(message, " ("_s, sqliteCode, ' ', unsafeSpan(sqliteMessage), ')'));
    }

    unsigned code() const { return m_code; }
    String messageIsolatedCopy() const { return m_message.isolatedCopy(); }

    enum SQLErrorCode {
        UNKNOWN_ERR = 0,
        DATABASE_ERR = 1,
        VERSION_ERR = 2,
        TOO_LARGE_ERR = 3,
        QUOTA_ERR = 4,
        SYNTAX_ERR = 5,
        CONSTRAINT_ERR = 6,
        TIMEOUT_ERR = 7
    };

private:
    SQLError(unsigned code, String&& message) : m_code(code), m_message(WTFMove(message).isolatedCopy()) { }
    unsigned m_code;
    String m_message;
};

} // namespace WebCore
