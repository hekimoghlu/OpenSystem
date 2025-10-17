/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

#include "ExceptionOr.h"
#include "URLPatternComponent.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScriptExecutionContext;
struct URLPatternInit;
struct URLPatternOptions;
struct URLPatternResult;
enum class BaseURLStringType : bool { Pattern, URL };

namespace URLPatternUtilities {
class URLPatternComponent;
}

class URLPattern final : public RefCounted<URLPattern> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(URLPattern);
public:
    using URLPatternInput = std::variant<String, URLPatternInit>;

    static ExceptionOr<Ref<URLPattern>> create(ScriptExecutionContext&, URLPatternInput&&, String&& baseURL, URLPatternOptions&&);
    static ExceptionOr<Ref<URLPattern>> create(ScriptExecutionContext&, std::optional<URLPatternInput>&&, URLPatternOptions&&);
    ~URLPattern();

    ExceptionOr<bool> test(ScriptExecutionContext&, std::optional<URLPatternInput>&&, String&& baseURL) const;

    ExceptionOr<std::optional<URLPatternResult>> exec(ScriptExecutionContext&, std::optional<URLPatternInput>&&, String&& baseURL) const;

    const String& protocol() const { return m_protocolComponent.patternString(); }
    const String& username() const { return m_usernameComponent.patternString(); }
    const String& password() const { return m_passwordComponent.patternString(); }
    const String& hostname() const { return m_hostnameComponent.patternString(); }
    const String& port() const { return m_portComponent.patternString(); }
    const String& pathname() const { return m_pathnameComponent.patternString(); }
    const String& search() const { return m_searchComponent.patternString(); }
    const String& hash() const { return m_hashComponent.patternString(); }

    bool hasRegExpGroups() const;

private:
    URLPattern();
    ExceptionOr<void> compileAllComponents(ScriptExecutionContext&, URLPatternInit&&, const URLPatternOptions&);
    ExceptionOr<std::optional<URLPatternResult>> match(ScriptExecutionContext&, std::variant<URL, URLPatternInput>&&, String&& baseURLString) const;

    URLPatternUtilities::URLPatternComponent m_protocolComponent;
    URLPatternUtilities::URLPatternComponent m_usernameComponent;
    URLPatternUtilities::URLPatternComponent m_passwordComponent;
    URLPatternUtilities::URLPatternComponent m_hostnameComponent;
    URLPatternUtilities::URLPatternComponent m_pathnameComponent;
    URLPatternUtilities::URLPatternComponent m_portComponent;
    URLPatternUtilities::URLPatternComponent m_searchComponent;
    URLPatternUtilities::URLPatternComponent m_hashComponent;
};

}
