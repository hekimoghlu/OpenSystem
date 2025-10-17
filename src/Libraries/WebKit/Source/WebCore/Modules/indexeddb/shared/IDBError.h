/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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

#include "DOMException.h"
#include "ExceptionCode.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class IDBError {
public:
    WEBCORE_EXPORT explicit IDBError(std::optional<ExceptionCode> = std::nullopt, const String& message = { });

    static IDBError userDeleteError()
    {
        return IDBError { ExceptionCode::UnknownError, "Database deleted by request of the user"_s };
    }
    
    static IDBError serverConnectionLostError()
    {
        return IDBError { ExceptionCode::UnknownError, "Connection to Indexed Database server lost. Refresh the page to try again"_s };
    }

    RefPtr<DOMException> toDOMException() const;

    std::optional<ExceptionCode> code() const { return m_code; }
    String name() const;
    String message() const;
    const String& messageForSerialization() const { return m_message; }

    bool isNull() const { return !m_code; }
    operator bool() const { return !isNull(); }

    IDBError isolatedCopy() const & { return IDBError { m_code, m_message.isolatedCopy() }; }
    IDBError isolatedCopy() && { return IDBError { m_code, WTFMove(m_message).isolatedCopy() }; }

private:
    std::optional<ExceptionCode> m_code;
    String m_message;
};

} // namespace WebCore
