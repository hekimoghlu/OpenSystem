/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

#include "IntlNumberFormat.h"
#include <unicode/unum.h>
#include <wtf/unicode/icu/ICUHelpers.h>

struct UPluralRules;

namespace JSC {

struct UPluralRulesDeleter {
    JS_EXPORT_PRIVATE void operator()(UPluralRules*);
};

enum class RelevantExtensionKey : uint8_t;

class IntlPluralRules final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlPluralRules*>(cell)->IntlPluralRules::~IntlPluralRules();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlPluralRulesSpace<mode>();
    }

    static IntlPluralRules* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    template<typename IntlType>
    friend void setNumberFormatDigitOptions(JSGlobalObject*, IntlType*, JSObject*, unsigned minimumFractionDigitsDefault, unsigned maximumFractionDigitsDefault, IntlNotation);
    template<typename IntlType>
    friend void appendNumberFormatDigitOptionsToSkeleton(IntlType*, StringBuilder&);

    void initializePluralRules(JSGlobalObject*, JSValue locales, JSValue options);
    JSValue select(JSGlobalObject*, double value) const;
    JSValue selectRange(JSGlobalObject*, double start, double end) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

private:
    IntlPluralRules(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    static Vector<String> localeData(const String&, RelevantExtensionKey);

    enum class Type : bool { Cardinal, Ordinal };

    std::unique_ptr<UPluralRules, UPluralRulesDeleter> m_pluralRules;
    std::unique_ptr<UNumberFormatter, UNumberFormatterDeleter> m_numberFormatter;
    std::unique_ptr<UNumberRangeFormatter, UNumberRangeFormatterDeleter> m_numberRangeFormatter;

    String m_locale;
    unsigned m_minimumIntegerDigits { 1 };
    unsigned m_minimumFractionDigits { 0 };
    unsigned m_maximumFractionDigits { 3 };
    unsigned m_minimumSignificantDigits { 0 };
    unsigned m_maximumSignificantDigits { 0 };
    unsigned m_roundingIncrement { 1 };
    IntlTrailingZeroDisplay m_trailingZeroDisplay { IntlTrailingZeroDisplay::Auto };
    RoundingMode m_roundingMode { RoundingMode::HalfExpand };
    IntlRoundingType m_roundingType { IntlRoundingType::FractionDigits };
    Type m_type { Type::Cardinal };
};

} // namespace JSC
