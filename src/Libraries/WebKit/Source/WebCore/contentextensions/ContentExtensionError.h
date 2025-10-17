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
#pragma once

#if ENABLE(CONTENT_EXTENSIONS)

#include <system_error>

#include <wtf/Forward.h>

namespace WebCore {
namespace ContentExtensions {

enum class ContentExtensionError {
    // JSON parser error
    JSONInvalid = 1,
    
    // JSON semantics error
    JSONTopLevelStructureNotAnArray,
    JSONInvalidObjectInTopLevelArray,
    JSONInvalidRule,
    JSONContainsNoRules,
    
    JSONInvalidTrigger,
    JSONInvalidURLFilterInTrigger,
    JSONInvalidTriggerFlagsArray,
    JSONInvalidStringInTriggerFlagsArray,
    JSONInvalidConditionList,
    JSONDomainNotLowerCaseASCII,
    JSONMultipleConditions,
    JSONTooManyRules,
    
    JSONInvalidAction,
    JSONInvalidActionType,
    JSONInvalidCSSDisplayNoneActionType,
    JSONInvalidNotification,
    JSONInvalidRegex,

    JSONRedirectMissing,
    JSONRedirectExtensionPathDoesNotStartWithSlash,
    JSONRedirectURLSchemeInvalid,
    JSONRedirectToJavaScriptURL,
    JSONRedirectURLInvalid,
    JSONRedirectInvalidType,
    JSONRedirectInvalidPort,
    JSONRedirectInvalidQuery,
    JSONRedirectInvalidFragment,

    JSONRemoveParametersNotStringArray,

    JSONAddOrReplaceParametersNotArray,
    JSONAddOrReplaceParametersKeyValueNotADictionary,
    JSONAddOrReplaceParametersKeyValueMissingKeyString,
    JSONAddOrReplaceParametersKeyValueMissingValueString,

    JSONModifyHeadersNotArray,
    JSONModifyHeadersInfoNotADictionary,
    JSONModifyHeadersMissingOperation,
    JSONModifyHeadersInvalidOperation,
    JSONModifyHeadersMissingHeader,
    JSONModifyHeadersMissingValue,
    JSONModifyHeadersInvalidPriority,

    ErrorWritingSerializedNFA,
};

extern ASCIILiteral WebKitContentBlockerDomain;
    
WEBCORE_EXPORT const std::error_category& contentExtensionErrorCategory();

inline std::error_code make_error_code(ContentExtensionError error)
{
    return { static_cast<int>(error), contentExtensionErrorCategory() };
}

} // namespace ContentExtensions
} // namespace WebCore

namespace std {
    template<> struct is_error_code_enum<WebCore::ContentExtensions::ContentExtensionError> : public true_type { };
}

#endif // ENABLE(CONTENT_EXTENSIONS)
