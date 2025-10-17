/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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

#include "Error.h"
#include "ErrorHandlingScope.h"
#include "ExceptionHelpers.h"
#include "JSGlobalObject.h"
#include "ParserTokens.h"
#include "SourceCode.h"
#include <wtf/text/WTFString.h>

namespace JSC {

class ParserError {
public:
    enum SyntaxErrorType : uint8_t {
        SyntaxErrorNone,
        SyntaxErrorIrrecoverable,
        SyntaxErrorUnterminatedLiteral,
        SyntaxErrorRecoverable
    };

    enum ErrorType : uint8_t {
        ErrorNone,
        StackOverflow,
        EvalError,
        OutOfMemory,
        SyntaxError
    };

    ParserError()
        : m_type(ErrorNone)
        , m_syntaxErrorType(SyntaxErrorNone)
    {
    }
    
    explicit ParserError(ErrorType type)
        : m_type(type)
        , m_syntaxErrorType(SyntaxErrorNone)
    {
    }

    ParserError(ErrorType type, SyntaxErrorType syntaxError, JSToken token)
        : m_token(token)
        , m_type(type)
        , m_syntaxErrorType(syntaxError)
    {
    }

    ParserError(ErrorType type, SyntaxErrorType syntaxError, JSToken token, const String& msg, int line)
        : m_token(token)
        , m_message(msg)
        , m_line(line)
        , m_type(type)
        , m_syntaxErrorType(syntaxError)
    {
    }

    bool isValid() const { return m_type != ErrorNone; }
    SyntaxErrorType syntaxErrorType() const { return m_syntaxErrorType; }
    const JSToken& token() const { return m_token; }
    const String& message() const { return m_message; }
    int line() const { return m_line; }
    ErrorType type() const { return m_type; }

    JSObject* toErrorObject(
        JSGlobalObject* globalObject,
        SourceCode source, // Note: We must copy the source here, since the objects that pass in their SourceCode field may be destroyed in addErrorInfo.
        int overrideLineNumber = -1)
    {
        switch (m_type) {
        case ErrorNone:
            return nullptr;
        case SyntaxError:
            return addErrorInfo(
                globalObject->vm(), 
                createSyntaxError(globalObject, m_message), 
                overrideLineNumber == -1 ? m_line : overrideLineNumber, source);
        case EvalError:
            return createSyntaxError(globalObject, m_message);
        case StackOverflow: {
            ErrorHandlingScope errorScope(getVM(globalObject));
            return createStackOverflowError(globalObject);
        }
        case OutOfMemory:
            return createOutOfMemoryError(globalObject);
        }
        CRASH();
        return nullptr;
    }

private:
    JSToken m_token;
    String m_message;
    int m_line { -1 };
    ErrorType m_type;
    SyntaxErrorType m_syntaxErrorType;
};

} // namespace JSC

namespace WTF {

inline void printInternal(PrintStream& out, JSC::ParserError::SyntaxErrorType type)
{
    switch (type) {
    case JSC::ParserError::SyntaxErrorNone:
        out.print("SyntaxErrorNone");
        return;
    case JSC::ParserError::SyntaxErrorIrrecoverable:
        out.print("SyntaxErrorIrrecoverable");
        return;
    case JSC::ParserError::SyntaxErrorUnterminatedLiteral:
        out.print("SyntaxErrorUnterminatedLiteral");
        return;
    case JSC::ParserError::SyntaxErrorRecoverable:
        out.print("SyntaxErrorRecoverable");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

inline void printInternal(PrintStream& out, JSC::ParserError::ErrorType type)
{
    switch (type) {
    case JSC::ParserError::ErrorNone:
        out.print("ErrorNone");
        return;
    case JSC::ParserError::StackOverflow:
        out.print("StackOverflow");
        return;
    case JSC::ParserError::EvalError:
        out.print("EvalError");
        return;
    case JSC::ParserError::OutOfMemory:
        out.print("OutOfMemory");
        return;
    case JSC::ParserError::SyntaxError:
        out.print("SyntaxError");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF
