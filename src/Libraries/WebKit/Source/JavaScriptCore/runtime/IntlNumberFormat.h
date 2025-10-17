/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include "MathCommon.h"
#include "TemporalObject.h"
#include <unicode/unum.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/unicode/icu/ICUHelpers.h>

struct UFormattedValue;
struct UNumberFormatter;
struct UNumberRangeFormatter;

namespace JSC {

class IntlFieldIterator;
class JSBoundFunction;
enum class RelevantExtensionKey : uint8_t;

enum class IntlRoundingType : uint8_t { FractionDigits, SignificantDigits, MorePrecision, LessPrecision };
enum class IntlRoundingPriority : uint8_t { Auto, MorePrecision, LessPrecision };
enum class IntlTrailingZeroDisplay : uint8_t { Auto, StripIfInteger };
enum class IntlNotation : uint8_t { Standard, Scientific, Engineering, Compact };
template<typename IntlType> void setNumberFormatDigitOptions(JSGlobalObject*, IntlType*, JSObject*, unsigned minimumFractionDigitsDefault, unsigned maximumFractionDigitsDefault, IntlNotation);
template<typename IntlType> void appendNumberFormatDigitOptionsToSkeleton(IntlType*, StringBuilder&);

struct UNumberFormatterDeleter {
    JS_EXPORT_PRIVATE void operator()(UNumberFormatter*);
};

struct UNumberRangeFormatterDeleter {
    JS_EXPORT_PRIVATE void operator()(UNumberRangeFormatter*);
};

class IntlMathematicalValue {
    WTF_MAKE_TZONE_ALLOCATED(IntlMathematicalValue);
public:
    enum class NumberType { Integer, Infinity, NaN, };
    using Value = std::variant<double, CString>;

    IntlMathematicalValue() = default;

    explicit IntlMathematicalValue(double value)
        : m_value(purifyNaN(value))
        , m_numberType(numberTypeFromDouble(value))
        , m_sign(!std::isnan(value) && std::signbit(value))
    { }

    explicit IntlMathematicalValue(NumberType numberType, bool sign, CString value)
        : m_value(value)
        , m_numberType(numberType)
        , m_sign(sign)
    {
    }

    static IntlMathematicalValue parseString(JSGlobalObject*, StringView);

    void ensureNonDouble()
    {
        if (std::holds_alternative<double>(m_value)) {
            switch (m_numberType) {
            case NumberType::Integer: {
                double value = std::get<double>(m_value);
                if (isNegativeZero(value))
                    m_value = CString("-0");
                else
                    m_value = String::number(value).ascii();
                break;
            }
            case NumberType::NaN:
                m_value = CString("nan");
                break;
            case NumberType::Infinity:
                m_value = CString(m_sign ? "-infinity" : "infinity");
                break;
            }
        }
    }

    NumberType numberType() const { return m_numberType; }
    bool sign() const { return m_sign; }
    std::optional<double> tryGetDouble() const
    {
        if (std::holds_alternative<double>(m_value))
            return std::get<double>(m_value);
        return std::nullopt;
    }
    const CString& getString() const
    {
        ASSERT(std::holds_alternative<CString>(m_value));
        return std::get<CString>(m_value);
    }

    static NumberType numberTypeFromDouble(double value)
    {
        if (std::isnan(value))
            return NumberType::NaN;
        if (!std::isfinite(value))
            return NumberType::Infinity;
        return NumberType::Integer;
    }

private:
    Value m_value { 0.0 };
    NumberType m_numberType { NumberType::Integer };
    bool m_sign { false };
};

class IntlNumberFormat final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlNumberFormat*>(cell)->IntlNumberFormat::~IntlNumberFormat();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlNumberFormatSpace<mode>();
    }

    static IntlNumberFormat* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeNumberFormat(JSGlobalObject*, JSValue locales, JSValue optionsValue);
    JSValue format(JSGlobalObject*, double) const;
    JSValue format(JSGlobalObject*, IntlMathematicalValue&&) const;
    JSValue formatToParts(JSGlobalObject*, double, JSString* sourceType = nullptr) const;
    JSValue formatToParts(JSGlobalObject*, IntlMathematicalValue&&, JSString* sourceType = nullptr) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

    JSValue formatRange(JSGlobalObject*, double, double) const;
    JSValue formatRange(JSGlobalObject*, IntlMathematicalValue&&, IntlMathematicalValue&&) const;

    JSValue formatRangeToParts(JSGlobalObject*, double, double) const;
    JSValue formatRangeToParts(JSGlobalObject*, IntlMathematicalValue&&, IntlMathematicalValue&&) const;

    JSBoundFunction* boundFormat() const { return m_boundFormat.get(); }
    void setBoundFormat(VM&, JSBoundFunction*);

    enum class Style : uint8_t { Decimal, Percent, Currency, Unit };

    static void formatToPartsInternal(JSGlobalObject*, Style, bool sign, IntlMathematicalValue::NumberType, const String& formatted, IntlFieldIterator&, JSArray*, JSString* sourceType, JSString* unit);
    static void formatRangeToPartsInternal(JSGlobalObject*, Style, IntlMathematicalValue&&, IntlMathematicalValue&&, const UFormattedValue*, JSArray*);

    template<typename IntlType>
    friend void setNumberFormatDigitOptions(JSGlobalObject*, IntlType*, JSObject*, unsigned minimumFractionDigitsDefault, unsigned maximumFractionDigitsDefault, IntlNotation);
    template<typename IntlType>
    friend void appendNumberFormatDigitOptionsToSkeleton(IntlType*, StringBuilder&);

    static ASCIILiteral notationString(IntlNotation);

    static IntlNumberFormat* unwrapForOldFunctions(JSGlobalObject*, JSValue);

    static ASCIILiteral roundingModeString(RoundingMode);
    static ASCIILiteral roundingPriorityString(IntlRoundingType);

private:
    IntlNumberFormat(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    static Vector<String> localeData(const String&, RelevantExtensionKey);

    enum class CurrencyDisplay : uint8_t { Code, Symbol, FormalSymbol, NarrowSymbol, Name, Never };
    enum class CurrencySign : uint8_t { Standard, Accounting };
    enum class UnitDisplay : uint8_t { Short, Narrow, Long };
    enum class CompactDisplay : uint8_t { Short, Long };
    enum class SignDisplay : uint8_t { Auto, Never, Always, ExceptZero, Negative };
    enum class UseGrouping : uint8_t { False, Min2, Auto, Always };

    static ASCIILiteral styleString(Style);
    static ASCIILiteral currencyDisplayString(CurrencyDisplay);
    static ASCIILiteral currencySignString(CurrencySign);
    static ASCIILiteral unitDisplayString(UnitDisplay);
    static ASCIILiteral compactDisplayString(CompactDisplay);
    static ASCIILiteral signDisplayString(SignDisplay);
    static ASCIILiteral trailingZeroDisplayString(IntlTrailingZeroDisplay);
    static JSValue useGroupingValue(VM&, UseGrouping);

    WriteBarrier<JSBoundFunction> m_boundFormat;
    std::unique_ptr<UNumberFormatter, UNumberFormatterDeleter> m_numberFormatter;
    std::unique_ptr<UNumberRangeFormatter, UNumberRangeFormatterDeleter> m_numberRangeFormatter;

    String m_locale;
    String m_numberingSystem;
    String m_currency;
    String m_unit;
    unsigned m_minimumIntegerDigits { 1 };
    unsigned m_minimumFractionDigits { 0 };
    unsigned m_maximumFractionDigits { 3 };
    unsigned m_minimumSignificantDigits { 0 };
    unsigned m_maximumSignificantDigits { 0 };
    unsigned m_roundingIncrement { 1 };
    Style m_style { Style::Decimal };
    CurrencyDisplay m_currencyDisplay;
    CurrencySign m_currencySign;
    UnitDisplay m_unitDisplay;
    CompactDisplay m_compactDisplay;
    IntlNotation m_notation { IntlNotation::Standard };
    SignDisplay m_signDisplay;
    IntlTrailingZeroDisplay m_trailingZeroDisplay { IntlTrailingZeroDisplay::Auto };
    UseGrouping m_useGrouping { UseGrouping::Always };
    RoundingMode m_roundingMode { RoundingMode::HalfExpand };
    IntlRoundingType m_roundingType { IntlRoundingType::FractionDigits };
};

} // namespace JSC
