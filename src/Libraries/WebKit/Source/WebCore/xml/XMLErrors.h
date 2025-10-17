/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/parser.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextPosition.h>

namespace WebCore {

class Document;

class XMLErrors {
    WTF_MAKE_TZONE_ALLOCATED(XMLErrors);
public:
    explicit XMLErrors(Document&);

    enum class Type : uint8_t { Warning, NonFatal, Fatal };
    void handleError(Type, const char* message, int lineNumber, int columnNumber);
    void handleError(Type, const char* message, TextPosition);

    void insertErrorMessageBlock();

private:
    void appendErrorMessage(ASCIILiteral typeString, TextPosition, const char* message);

    CheckedRef<Document> m_document;
    int m_errorCount { 0 };
    std::optional<TextPosition> m_lastErrorPosition;
    StringBuilder m_errorMessages;
};

} // namespace WebCore
