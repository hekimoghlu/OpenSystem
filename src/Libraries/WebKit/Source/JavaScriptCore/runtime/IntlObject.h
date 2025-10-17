/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#include "JSCJSValueInlines.h"
#include "JSObject.h"
#include <wtf/RobinHoodHashSet.h>

struct UFieldPositionIterator;

namespace JSC {

extern const uint8_t ducetLevel1Weights[256];
extern const uint8_t ducetLevel3Weights[256];

enum class LocaleMatcher : uint8_t {
    Lookup,
    BestFit,
};

#define JSC_INTL_RELEVANT_EXTENSION_KEYS(macro) \
    macro(ca, Ca) /* calendar */ \
    macro(co, Co) /* collation */ \
    macro(hc, Hc) /* hour-cycle */ \
    macro(kf, Kf) /* case-first */ \
    macro(kn, Kn) /* numeric */ \
    macro(nu, Nu) /* numbering-system */ \

enum class RelevantExtensionKey : uint8_t {
#define JSC_DECLARE_INTL_RELEVANT_EXTENSION_KEYS(lowerName, capitalizedName) capitalizedName,
JSC_INTL_RELEVANT_EXTENSION_KEYS(JSC_DECLARE_INTL_RELEVANT_EXTENSION_KEYS)
#undef JSC_DECLARE_INTL_RELEVANT_EXTENSION_KEYS
};
#define JSC_COUNT_INTL_RELEVANT_EXTENSION_KEYS(lowerName, capitalizedName) + 1
static constexpr uint8_t numberOfRelevantExtensionKeys = 0 JSC_INTL_RELEVANT_EXTENSION_KEYS(JSC_COUNT_INTL_RELEVANT_EXTENSION_KEYS);
#undef JSC_COUNT_INTL_RELEVANT_EXTENSION_KEYS

struct MeasureUnit {
    ASCIILiteral type;
    ASCIILiteral subType;
};

extern JS_EXPORT_PRIVATE const MeasureUnit simpleUnits[45];

class IntlObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(IntlObject, Base);
        return &vm.plainObjectSpace();
    }

    static IntlObject* create(VM&, JSGlobalObject*, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    IntlObject(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

String defaultLocale(JSGlobalObject*);
using LocaleSet = MemoryCompactLookupOnlyRobinHoodHashSet<String>;
const LocaleSet& intlAvailableLocales();
const LocaleSet& intlCollatorAvailableLocales();
const LocaleSet& intlSegmenterAvailableLocales();
inline const LocaleSet& intlDateTimeFormatAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlDisplayNamesAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlNumberFormatAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlPluralRulesAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlRelativeTimeFormatAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlListFormatAvailableLocales() { return intlAvailableLocales(); }
inline const LocaleSet& intlDurationFormatAvailableLocales() { return intlAvailableLocales(); }

using CalendarID = unsigned;
const Vector<String>& intlAvailableCalendars();

extern CalendarID iso8601CalendarIDStorage;
CalendarID iso8601CalendarIDSlow();
inline CalendarID iso8601CalendarID()
{
    unsigned value = iso8601CalendarIDStorage;
    if (value == std::numeric_limits<CalendarID>::max())
        return iso8601CalendarIDSlow();
    return value;
}

using TimeZoneID = unsigned;
const Vector<String>& intlAvailableTimeZones();

extern TimeZoneID utcTimeZoneIDStorage;
TimeZoneID utcTimeZoneIDSlow();
CalendarID utcTimeZoneID();

TriState intlBooleanOption(JSGlobalObject*, JSObject* options, PropertyName);
String intlStringOption(JSGlobalObject*, JSObject* options, PropertyName, std::initializer_list<ASCIILiteral> values, ASCIILiteral notFound, ASCIILiteral fallback);
unsigned intlNumberOption(JSGlobalObject*, JSObject* options, PropertyName, unsigned minimum, unsigned maximum, unsigned fallback);
unsigned intlDefaultNumberOption(JSGlobalObject*, JSValue, PropertyName, unsigned minimum, unsigned maximum, unsigned fallback);
Vector<char, 32> localeIDBufferForLanguageTagWithNullTerminator(const CString&);
String languageTagForLocaleID(const char*, bool isImmortal = false);
Vector<String> canonicalizeLocaleList(JSGlobalObject*, JSValue locales);

using ResolveLocaleOptions = std::array<std::optional<String>, numberOfRelevantExtensionKeys>;
using RelevantExtensions = std::array<String, numberOfRelevantExtensionKeys>;
struct ResolvedLocale {
    String locale;
    String dataLocale;
    RelevantExtensions extensions;
};

ResolvedLocale resolveLocale(JSGlobalObject*, const LocaleSet& availableLocales, const Vector<String>& requestedLocales, LocaleMatcher, const ResolveLocaleOptions&, std::initializer_list<RelevantExtensionKey> relevantExtensionKeys, Vector<String> (*localeData)(const String&, RelevantExtensionKey));
JSValue supportedLocales(JSGlobalObject*, const LocaleSet& availableLocales, const Vector<String>& requestedLocales, JSValue options);
String removeUnicodeLocaleExtension(const String& locale);
String bestAvailableLocale(const LocaleSet& availableLocales, const String& requestedLocale);
template<typename Predicate> String bestAvailableLocale(const String& requestedLocale, Predicate);
Vector<String> numberingSystemsForLocale(const String& locale);

Vector<char, 32> canonicalizeUnicodeExtensionsAfterICULocaleCanonicalization(Vector<char, 32>&&);

bool isUnicodeLocaleIdentifierType(StringView);

bool isUnicodeLanguageSubtag(StringView);
bool isUnicodeScriptSubtag(StringView);
bool isUnicodeRegionSubtag(StringView);
bool isUnicodeVariantSubtag(StringView);
bool isUnicodeLanguageId(StringView);
bool isStructurallyValidLanguageTag(StringView);
String canonicalizeUnicodeLocaleID(const CString& languageTag);

bool isWellFormedCurrencyCode(StringView);

std::optional<Vector<char, 32>> canonicalizeLocaleIDWithoutNullTerminator(const char* localeID);

struct UFieldPositionIteratorDeleter {
    void operator()(UFieldPositionIterator*) const;
};

std::optional<String> mapICUCollationKeywordToBCP47(const String&);
std::optional<String> mapICUCalendarKeywordToBCP47(const String&);
std::optional<String> mapBCP47ToICUCalendarKeyword(const String&);


inline CalendarID utcTimeZoneID()
{
    unsigned value = utcTimeZoneIDStorage;
    if (value == std::numeric_limits<TimeZoneID>::max())
        return utcTimeZoneIDSlow();
    return value;
}

} // namespace JSC
