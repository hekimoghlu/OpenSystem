/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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

#include "JSObject.h"
#include <unicode/uldnames.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

class IntlDisplayNames final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlDisplayNames*>(cell)->IntlDisplayNames::~IntlDisplayNames();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlDisplayNamesSpace<mode>();
    }

    static IntlDisplayNames* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeDisplayNames(JSGlobalObject*, JSValue localesValue, JSValue optionsValue);

    JSValue of(JSGlobalObject*, JSValue) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

private:
    IntlDisplayNames(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;

    enum class Style : uint8_t { Narrow, Short, Long };
    enum class Type : uint8_t { Language, Region, Script, Currency, Calendar, DateTimeField };
    enum class Fallback : uint8_t { Code, None };
    enum class LanguageDisplay : uint8_t { Dialect, Standard };

    static ASCIILiteral styleString(Style);
    static ASCIILiteral typeString(Type);
    static ASCIILiteral fallbackString(Fallback);
    static ASCIILiteral languageDisplayString(LanguageDisplay);

    using ULocaleDisplayNamesDeleter = ICUDeleter<uldn_close>;
    std::unique_ptr<ULocaleDisplayNames, ULocaleDisplayNamesDeleter> m_displayNames;
    String m_locale;
    // FIXME: We should store it only when m_type is Currency.
    // https://bugs.webkit.org/show_bug.cgi?id=213773
    CString m_localeCString;
    Style m_style { Style::Long };
    Type m_type { Type::Language };
    Fallback m_fallback { Fallback::Code };
    LanguageDisplay m_languageDisplay { LanguageDisplay::Dialect };
};

} // namespace JSC
