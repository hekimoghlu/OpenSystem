/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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

#include <wtf/Forward.h>
#include <wtf/JSONValues.h>
#include <wtf/text/WTFString.h>

namespace WebDriver {

class CommandResult {
public:
    // Â§6.6 Handling Errors.
    // https://www.w3.org/TR/webdriver/#handling-errors
    enum class ErrorCode {
        ElementClickIntercepted,
        ElementNotSelectable,
        ElementNotInteractable,
        DetachedShadowRoot,
        InvalidArgument,
        InvalidElementState,
        InvalidSelector,
        InvalidSessionID,
        JavascriptError,
        MoveTargetOutOfBounds,
        NoSuchAlert,
        NoSuchCookie,
        NoSuchElement,
        NoSuchFrame,
        NoSuchShadowRoot,
        NoSuchWindow,
        ScriptTimeout,
        SessionNotCreated,
        StaleElementReference,
        Timeout,
        UnableToCaptureScreen,
        UnexpectedAlertOpen,
        UnknownCommand,
        UnknownError,
        UnsupportedOperation,
    };

    static CommandResult success(RefPtr<JSON::Value>&& result = nullptr)
    {
        return CommandResult(WTFMove(result));
    }

    static CommandResult fail(RefPtr<JSON::Value>&& result = nullptr)
    {
        return CommandResult(WTFMove(result), CommandResult::ErrorCode::UnknownError);
    }

    static CommandResult fail(ErrorCode errorCode, std::optional<String> errorMessage = std::nullopt)
    {
        return CommandResult(errorCode, errorMessage);
    }

    unsigned httpStatusCode() const;
    static unsigned errorCodeToHTTPStatusCode(ErrorCode);
    const RefPtr<JSON::Value>& result() const { return m_result; };
    void setAdditionalErrorData(RefPtr<JSON::Object>&& errorData) { m_errorAdditionalData = WTFMove(errorData); }
    bool isError() const { return !!m_errorCode; }
    ErrorCode errorCode() const { ASSERT(isError()); return m_errorCode.value(); }
    String errorString() const;
    std::optional<String> errorMessage() const { ASSERT(isError()); return m_errorMessage; }
    const RefPtr<JSON::Object>& additionalErrorData() const { return m_errorAdditionalData; }

private:
    explicit CommandResult(RefPtr<JSON::Value>&&, std::optional<ErrorCode> = std::nullopt);
    explicit CommandResult(ErrorCode, std::optional<String> = std::nullopt);

    RefPtr<JSON::Value> m_result;
    std::optional<ErrorCode> m_errorCode;
    std::optional<String> m_errorMessage;
    RefPtr<JSON::Object> m_errorAdditionalData;
};

} // namespace WebDriver
