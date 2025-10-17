/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#include "ISO8601.h"
#include "IntlListFormat.h"
#include "IntlNumberFormat.h"
#include "JSObject.h"
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

class IntlDurationFormat final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlDurationFormat*>(cell)->IntlDurationFormat::~IntlDurationFormat();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlDurationFormatSpace<mode>();
    }

    static IntlDurationFormat* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeDurationFormat(JSGlobalObject*, JSValue localesValue, JSValue optionsValue);

    JSValue format(JSGlobalObject*, ISO8601::Duration) const;
    JSValue formatToParts(JSGlobalObject*, ISO8601::Duration) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

    enum class Display : uint8_t { Always, Auto };
    enum class Style : uint8_t { Long, Short, Narrow, Digital };
    enum class UnitStyle : uint8_t { Long, Short, Narrow, Numeric, TwoDigit };
    static constexpr unsigned numberOfUnitStyles = 5;
    static_assert(static_cast<unsigned>(Style::Long) == static_cast<unsigned>(UnitStyle::Long));
    static_assert(static_cast<unsigned>(Style::Short) == static_cast<unsigned>(UnitStyle::Short));
    static_assert(static_cast<unsigned>(Style::Narrow) == static_cast<unsigned>(UnitStyle::Narrow));

    class UnitData {
    public:
        UnitData() = default;
        UnitData(UnitStyle style, Display display)
            : m_style(style)
            , m_display(display)
        {
        }

        UnitStyle style() const { return m_style; }
        Display display() const { return m_display; }

    private:
        UnitStyle m_style : 7 { UnitStyle::Long };
        Display m_display : 1 { Display::Always };
    };

    const UnitData* units() const { return m_units; }
    unsigned fractionalDigits() const { return m_fractionalDigits; }
    const String& numberingSystem() const { return m_numberingSystem; }
    const CString& dataLocaleWithExtensions() const { return m_dataLocaleWithExtensions; }

private:
    IntlDurationFormat(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;

    static ASCIILiteral styleString(Style);
    static ASCIILiteral unitStyleString(UnitStyle);
    static ASCIILiteral displayString(Display);

    std::unique_ptr<UListFormatter, UListFormatterDeleter> m_listFormat;
    String m_locale;
    String m_numberingSystem;
    CString m_dataLocaleWithExtensions;
    unsigned m_fractionalDigits { 0 };
    Style m_style { Style::Long };
    UnitData m_units[numberOfTemporalUnits] { };
};

} // namespace JSC
