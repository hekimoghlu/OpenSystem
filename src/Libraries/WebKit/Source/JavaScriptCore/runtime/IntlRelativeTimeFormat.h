/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include <unicode/ureldatefmt.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

class IntlRelativeTimeFormat final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlRelativeTimeFormat*>(cell)->IntlRelativeTimeFormat::~IntlRelativeTimeFormat();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlRelativeTimeFormatSpace<mode>();
    }

    static IntlRelativeTimeFormat* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeRelativeTimeFormat(JSGlobalObject*, JSValue locales, JSValue options);
    JSValue format(JSGlobalObject*, double, StringView unitString) const;
    JSValue formatToParts(JSGlobalObject*, double, StringView unitString) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

private:
    IntlRelativeTimeFormat(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    static Vector<String> localeData(const String&, RelevantExtensionKey);

    String formatInternal(JSGlobalObject*, double, StringView unit) const;

    enum class Style : uint8_t { Long, Short, Narrow };

    using URelativeDateTimeFormatterDeleter = ICUDeleter<ureldatefmt_close>;
    using UNumberFormatDeleter = ICUDeleter<unum_close>;

    static ASCIILiteral styleString(Style);

    std::unique_ptr<URelativeDateTimeFormatter, URelativeDateTimeFormatterDeleter> m_relativeDateTimeFormatter;
    std::unique_ptr<UNumberFormat, UNumberFormatDeleter> m_numberFormat;

    String m_locale;
    String m_numberingSystem;
    Style m_style { Style::Long };
    bool m_numeric { true };
};

} // namespace JSC
