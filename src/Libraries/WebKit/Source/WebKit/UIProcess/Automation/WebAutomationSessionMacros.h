/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#define errorNameAndDetailsSeparator ';'

// Make sure the predefined error name is valid, otherwise use InternalError.
#define VALIDATED_ERROR_MESSAGE(errorString) Inspector::Protocol::AutomationHelpers::parseEnumValueFromString<Inspector::Protocol::Automation::ErrorMessage>(errorString).value_or(Inspector::Protocol::Automation::ErrorMessage::InternalError)

// If the error name is incorrect for these macros, it will be a compile-time error.
#define STRING_FOR_PREDEFINED_ERROR_NAME(errorName) Inspector::Protocol::AutomationHelpers::getEnumConstantValue(Inspector::Protocol::Automation::ErrorMessage::errorName)
#define STRING_FOR_PREDEFINED_ERROR_NAME_AND_DETAILS(errorName, detailsString) makeString(Inspector::Protocol::AutomationHelpers::getEnumConstantValue(Inspector::Protocol::Automation::ErrorMessage::errorName), errorNameAndDetailsSeparator, detailsString)

// If the error message is not a predefined error, InternalError will be used instead.
#define STRING_FOR_PREDEFINED_ERROR_MESSAGE(errorMessage) Inspector::Protocol::AutomationHelpers::getEnumConstantValue(VALIDATED_ERROR_MESSAGE(errorMessage))
#define STRING_FOR_PREDEFINED_ERROR_MESSAGE_AND_DETAILS(errorMessage, detailsString) makeString(Inspector::Protocol::AutomationHelpers::getEnumConstantValue(VALIDATED_ERROR_MESSAGE(errorMessage)), errorNameAndDetailsSeparator, detailsString)

#define AUTOMATION_COMMAND_ERROR_WITH_NAME(errorName) AutomationCommandError(Inspector::Protocol::Automation::ErrorMessage::errorName)
#define AUTOMATION_COMMAND_ERROR_WITH_MESSAGE(errorString) AutomationCommandError(VALIDATED_ERROR_MESSAGE(errorString))

// Convenience macros for filling in the error string of synchronous commands in bailout branches.
#define SYNC_FAIL_WITH_PREDEFINED_ERROR(errorName) \
do { \
    return makeUnexpected(STRING_FOR_PREDEFINED_ERROR_NAME(errorName)); \
} while (false)

#define SYNC_FAIL_WITH_PREDEFINED_ERROR_AND_DETAILS(errorName, detailsString) \
do { \
    return makeUnexpected(STRING_FOR_PREDEFINED_ERROR_NAME_AND_DETAILS(errorName, detailsString)); \
} while (false)

#define ASYNC_FAIL_WITH_PREDEFINED_ERROR(errorName) \
do { \
    callback->sendFailure(STRING_FOR_PREDEFINED_ERROR_NAME(errorName)); \
    return; \
} while (false)

#define ASYNC_FAIL_WITH_PREDEFINED_ERROR_AND_DETAILS(errorName, detailsString) \
do { \
    callback->sendFailure(STRING_FOR_PREDEFINED_ERROR_NAME_AND_DETAILS(errorName, detailsString)); \
    return; \
} while (false)
