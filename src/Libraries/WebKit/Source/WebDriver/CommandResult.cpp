/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#include "CommandResult.h"

namespace WebDriver {

// These error codes are specified in JSON-RPC 2.0, Section 5.1.
enum ProtocolErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    ServerError = -32000
};

CommandResult::CommandResult(RefPtr<JSON::Value>&& result, std::optional<ErrorCode> errorCode)
    : m_errorCode(errorCode)
{
    if (!m_errorCode) {
        m_result = WTFMove(result);
        return;
    }

    if (!result)
        return;

    auto errorObject = result->asObject();
    if (!errorObject)
        return;

    auto error = errorObject->getInteger("code"_s);
    if (!error)
        return;

    auto errorMessage = errorObject->getString("message"_s);
    if (!errorMessage)
        return;

    switch (*error) {
    case ProtocolErrorCode::ParseError:
    case ProtocolErrorCode::InvalidRequest:
    case ProtocolErrorCode::MethodNotFound:
    case ProtocolErrorCode::InvalidParams:
    case ProtocolErrorCode::InternalError:
        m_errorCode = ErrorCode::UnknownError;
        m_errorMessage = errorMessage;
        break;
    case ProtocolErrorCode::ServerError: {
        String errorName;
        auto position = errorMessage.find(';');
        if (position != notFound) {
            errorName = errorMessage.left(position);
            m_errorMessage = errorMessage.substring(position + 1);
        } else
            errorName = errorMessage;

        if (errorName == "WindowNotFound"_s)
            m_errorCode = ErrorCode::NoSuchWindow;
        else if (errorName == "FrameNotFound"_s)
            m_errorCode = ErrorCode::NoSuchFrame;
        else if (errorName == "NotImplemented"_s)
            m_errorCode = ErrorCode::UnsupportedOperation;
        else if (errorName == "ElementNotInteractable"_s)
            m_errorCode = ErrorCode::ElementNotInteractable;
        else if (errorName == "JavaScriptError"_s)
            m_errorCode = ErrorCode::JavascriptError;
        else if (errorName == "JavaScriptTimeout"_s)
            m_errorCode = ErrorCode::ScriptTimeout;
        else if (errorName == "NodeNotFound"_s)
            m_errorCode = ErrorCode::StaleElementReference;
        else if (errorName == "InvalidNodeIdentifier"_s)
            m_errorCode = ErrorCode::NoSuchElement;
        else if (errorName == "MissingParameter"_s || errorName == "InvalidParameter"_s)
            m_errorCode = ErrorCode::InvalidArgument;
        else if (errorName == "InvalidElementState"_s)
            m_errorCode = ErrorCode::InvalidElementState;
        else if (errorName == "InvalidSelector"_s)
            m_errorCode = ErrorCode::InvalidSelector;
        else if (errorName == "Timeout"_s)
            m_errorCode = ErrorCode::Timeout;
        else if (errorName == "NoJavaScriptDialog"_s)
            m_errorCode = ErrorCode::NoSuchAlert;
        else if (errorName == "ElementNotSelectable"_s)
            m_errorCode = ErrorCode::ElementNotSelectable;
        else if (errorName == "ScreenshotError"_s)
            m_errorCode = ErrorCode::UnableToCaptureScreen;
        else if (errorName == "UnexpectedAlertOpen"_s)
            m_errorCode = ErrorCode::UnexpectedAlertOpen;
        else if (errorName == "TargetOutOfBounds"_s)
            m_errorCode = ErrorCode::MoveTargetOutOfBounds;

        break;
    }
    }
}

CommandResult::CommandResult(ErrorCode errorCode, std::optional<String> errorMessage)
    : m_errorCode(errorCode)
    , m_errorMessage(errorMessage)
{
}

unsigned CommandResult::httpStatusCode() const
{
    if (!m_errorCode)
        return 200;

    return errorCodeToHTTPStatusCode(m_errorCode.value());
}

unsigned CommandResult::errorCodeToHTTPStatusCode(ErrorCode errorCode)
{
    // Â§6.6 Handling Errors.
    // https://www.w3.org/TR/webdriver/#handling-errors
    switch (errorCode) {
    case ErrorCode::ElementClickIntercepted:
    case ErrorCode::ElementNotSelectable:
    case ErrorCode::ElementNotInteractable:
    case ErrorCode::InvalidArgument:
    case ErrorCode::InvalidElementState:
    case ErrorCode::InvalidSelector:
        return 400;
    case ErrorCode::NoSuchAlert:
    case ErrorCode::NoSuchCookie:
    case ErrorCode::NoSuchElement:
    case ErrorCode::NoSuchFrame:
    case ErrorCode::NoSuchWindow:
    case ErrorCode::NoSuchShadowRoot:
    case ErrorCode::StaleElementReference:
    case ErrorCode::DetachedShadowRoot:
    case ErrorCode::InvalidSessionID:
    case ErrorCode::UnknownCommand:
        return 404;
    case ErrorCode::JavascriptError:
    case ErrorCode::MoveTargetOutOfBounds:
    case ErrorCode::ScriptTimeout:
    case ErrorCode::SessionNotCreated:
    case ErrorCode::Timeout:
    case ErrorCode::UnableToCaptureScreen:
    case ErrorCode::UnexpectedAlertOpen:
    case ErrorCode::UnknownError:
    case ErrorCode::UnsupportedOperation:
        return 500;
    }

    ASSERT_NOT_REACHED();
    return 200;
}

String CommandResult::errorString() const
{
    ASSERT(isError());

    switch (m_errorCode.value()) {
    case ErrorCode::ElementClickIntercepted:
        return "element click intercepted"_s;
    case ErrorCode::ElementNotSelectable:
        return "element not selectable"_s;
    case ErrorCode::ElementNotInteractable:
        return "element not interactable"_s;
    case ErrorCode::DetachedShadowRoot:
        return "detached shadow root"_s;
    case ErrorCode::InvalidArgument:
        return "invalid argument"_s;
    case ErrorCode::InvalidElementState:
        return "invalid element state"_s;
    case ErrorCode::InvalidSelector:
        return "invalid selector"_s;
    case ErrorCode::InvalidSessionID:
        return "invalid session id"_s;
    case ErrorCode::JavascriptError:
        return "javascript error"_s;
    case ErrorCode::NoSuchAlert:
        return "no such alert"_s;
    case ErrorCode::NoSuchCookie:
        return "no such cookie"_s;
    case ErrorCode::NoSuchElement:
        return "no such element"_s;
    case ErrorCode::NoSuchFrame:
        return "no such frame"_s;
    case ErrorCode::NoSuchShadowRoot:
        return "no such shadow root"_s;
    case ErrorCode::NoSuchWindow:
        return "no such window"_s;
    case ErrorCode::ScriptTimeout:
        return "script timeout"_s;
    case ErrorCode::SessionNotCreated:
        return "session not created"_s;
    case ErrorCode::StaleElementReference:
        return "stale element reference"_s;
    case ErrorCode::Timeout:
        return "timeout"_s;
    case ErrorCode::UnableToCaptureScreen:
        return "unable to capture screen"_s;
    case ErrorCode::MoveTargetOutOfBounds:
        return "move target out of bounds"_s;
    case ErrorCode::UnexpectedAlertOpen:
        return "unexpected alert open"_s;
    case ErrorCode::UnknownCommand:
        return "unknown command"_s;
    case ErrorCode::UnknownError:
        return "unknown error"_s;
    case ErrorCode::UnsupportedOperation:
        return "unsupported operation"_s;
    }

    ASSERT_NOT_REACHED();
    return emptyString();
}

} // namespace WebDriver
