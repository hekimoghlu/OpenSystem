/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

#include "JSWebExtensionAPISidebarAction.h"
#include "WebExtensionAPIObject.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

using SidebarError = RetainPtr<NSString>;
// In this variant, `monostate` indicates that we have neither a window or tab identifier, but no error
using ParseResult = std::variant<std::monostate, WebExtensionTabIdentifier, WebExtensionWindowIdentifier, SidebarError>;

template<typename T, typename VARIANT_T>
struct isVariantMember;
template<typename T, typename... ALL_T>
struct isVariantMember<T, std::variant<ALL_T...>> : public std::disjunction<std::is_same<T, ALL_T>...> { };

template<typename OptType, typename... Types>
std::optional<OptType> toOptional(std::variant<Types...>& variant)
{
    if (std::holds_alternative<OptType>(variant))
        return WTFMove(std::get<OptType>(variant));
    return std::nullopt;
}

template<typename VariantType>
SidebarError indicatesError(const VariantType& variant)
{
    static_assert(isVariantMember<SidebarError, VariantType>::value);

    if (std::holds_alternative<SidebarError>(variant))
        return WTFMove(std::get<SidebarError>(variant));
    return nil;
}

class WebExtensionAPISidebarAction : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPISidebarAction, sidebarAction, sidebarAction);

public:
#if PLATFORM(COCOA)
    void open(Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void close(Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void toggle(Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void isOpen(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getPanel(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setPanel(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getTitle(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setTitle(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setIcon(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
#endif // PLATFORM(COCOA)
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
