/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "config.h"
#include "IntlLocalePrototype.h"

#include "IntlLocale.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncMaximize);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncMinimize);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncToString);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetCalendars);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetCollations);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetHourCycles);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetNumberingSystems);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetTimeZones);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetTextInfo);
static JSC_DECLARE_HOST_FUNCTION(intlLocalePrototypeFuncGetWeekInfo);

static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterBaseName);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterCalendar);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterCaseFirst);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterCollation);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterFirstDayOfWeek);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterHourCycle);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterNumeric);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterNumberingSystem);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterLanguage);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterScript);
static JSC_DECLARE_CUSTOM_GETTER(intlLocalePrototypeGetterRegion);

}

#include "IntlLocalePrototype.lut.h"

namespace JSC {

const ClassInfo IntlLocalePrototype::s_info = { "Intl.Locale"_s, &Base::s_info, &localePrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlLocalePrototype) };

/* Source for IntlLocalePrototype.lut.h
@begin localePrototypeTable
  maximize            intlLocalePrototypeFuncMaximize            DontEnum|Function 0
  minimize            intlLocalePrototypeFuncMinimize            DontEnum|Function 0
  toString            intlLocalePrototypeFuncToString            DontEnum|Function 0
  getCalendars        intlLocalePrototypeFuncGetCalendars        DontEnum|Function 0
  getCollations       intlLocalePrototypeFuncGetCollations       DontEnum|Function 0
  getHourCycles       intlLocalePrototypeFuncGetHourCycles       DontEnum|Function 0
  getNumberingSystems intlLocalePrototypeFuncGetNumberingSystems DontEnum|Function 0
  getTimeZones        intlLocalePrototypeFuncGetTimeZones        DontEnum|Function 0
  getTextInfo         intlLocalePrototypeFuncGetTextInfo         DontEnum|Function 0
  getWeekInfo         intlLocalePrototypeFuncGetWeekInfo         DontEnum|Function 0
  baseName            intlLocalePrototypeGetterBaseName          DontEnum|ReadOnly|CustomAccessor
  calendar            intlLocalePrototypeGetterCalendar          DontEnum|ReadOnly|CustomAccessor
  caseFirst           intlLocalePrototypeGetterCaseFirst         DontEnum|ReadOnly|CustomAccessor
  collation           intlLocalePrototypeGetterCollation         DontEnum|ReadOnly|CustomAccessor
  firstDayOfWeek      intlLocalePrototypeGetterFirstDayOfWeek    DontEnum|ReadOnly|CustomAccessor
  hourCycle           intlLocalePrototypeGetterHourCycle         DontEnum|ReadOnly|CustomAccessor
  numeric             intlLocalePrototypeGetterNumeric           DontEnum|ReadOnly|CustomAccessor
  numberingSystem     intlLocalePrototypeGetterNumberingSystem   DontEnum|ReadOnly|CustomAccessor
  language            intlLocalePrototypeGetterLanguage          DontEnum|ReadOnly|CustomAccessor
  script              intlLocalePrototypeGetterScript            DontEnum|ReadOnly|CustomAccessor
  region              intlLocalePrototypeGetterRegion            DontEnum|ReadOnly|CustomAccessor
@end
*/

IntlLocalePrototype* IntlLocalePrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlLocalePrototype>(vm)) IntlLocalePrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlLocalePrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlLocalePrototype::IntlLocalePrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlLocalePrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.maximize
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncMaximize, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.maximize called on value that's not a Locale"_s);

    IntlLocale* newLocale = IntlLocale::create(vm, globalObject->localeStructure());
    scope.release();
    newLocale->initializeLocale(globalObject, locale->maximal(), jsUndefined());
    return JSValue::encode(newLocale);
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.minimize
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncMinimize, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.minimize called on value that's not a Locale"_s);

    IntlLocale* newLocale = IntlLocale::create(vm, globalObject->localeStructure());
    scope.release();
    newLocale->initializeLocale(globalObject, locale->minimal(), jsUndefined());
    return JSValue::encode(newLocale);
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.toString
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.toString called on value that's not a Locale"_s);

    const String& fullString = locale->toString();
    RELEASE_AND_RETURN(scope, JSValue::encode(fullString.isEmpty() ? jsUndefined() : jsString(vm, fullString)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.baseName
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterBaseName, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.baseName called on value that's not a Locale"_s);

    const String& baseName = locale->baseName();
    RELEASE_AND_RETURN(scope, JSValue::encode(baseName.isEmpty() ? jsUndefined() : jsString(vm, baseName)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.calendar
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterCalendar, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.calendar called on value that's not a Locale"_s);

    const String& calendar = locale->calendar();
    RELEASE_AND_RETURN(scope, JSValue::encode(calendar.isNull() ? jsUndefined() : jsString(vm, calendar)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getCalendars
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetCalendars, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getCalendars called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->calendars(globalObject)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.caseFirst
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterCaseFirst, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.caseFirst called on value that's not a Locale"_s);

    const String& caseFirst = locale->caseFirst();
    RELEASE_AND_RETURN(scope, JSValue::encode(caseFirst.isNull() ? jsUndefined() : jsString(vm, caseFirst)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.collation
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterCollation, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.collation called on value that's not a Locale"_s);

    const String& collation = locale->collation();
    RELEASE_AND_RETURN(scope, JSValue::encode(collation.isNull() ? jsUndefined() : jsString(vm, collation)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getCollations
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetCollations, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getCollations called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->collations(globalObject)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.firstDayOfWeek
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterFirstDayOfWeek, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.firstDayOfWeek called on value that's not a Locale"_s);

    const String& firstDayOfWeek = locale->firstDayOfWeek();
    RELEASE_AND_RETURN(scope, JSValue::encode(firstDayOfWeek.isNull() ? jsUndefined() : jsString(vm, firstDayOfWeek)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.hourCycle
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterHourCycle, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.hourCycle called on value that's not a Locale"_s);

    const String& hourCycle = locale->hourCycle();
    RELEASE_AND_RETURN(scope, JSValue::encode(hourCycle.isNull() ? jsUndefined() : jsString(vm, hourCycle)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getHourCycles
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetHourCycles, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getHourCycles called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->hourCycles(globalObject)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.numeric
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterNumeric, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.numeric called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(jsBoolean(locale->numeric() == TriState::True)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.numberingSystem
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterNumberingSystem, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.numberingSystem called on value that's not a Locale"_s);

    const String& numberingSystem = locale->numberingSystem();
    RELEASE_AND_RETURN(scope, JSValue::encode(numberingSystem.isNull() ? jsUndefined() : jsString(vm, numberingSystem)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getNumberingSystems
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetNumberingSystems, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getNumberingSystems called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->numberingSystems(globalObject)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.language
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterLanguage, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.language called on value that's not a Locale"_s);

    const String& language = locale->language();
    RELEASE_AND_RETURN(scope, JSValue::encode(language.isEmpty() ? jsUndefined() : jsString(vm, language)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.script
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterScript, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.script called on value that's not a Locale"_s);

    const String& script = locale->script();
    RELEASE_AND_RETURN(scope, JSValue::encode(script.isEmpty() ? jsUndefined() : jsString(vm, script)));
}

// https://tc39.es/ecma402/#sec-Intl.Locale.prototype.region
JSC_DEFINE_CUSTOM_GETTER(intlLocalePrototypeGetterRegion, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(JSValue::decode(thisValue));
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.region called on value that's not a Locale"_s);

    const String& region = locale->region();
    RELEASE_AND_RETURN(scope, JSValue::encode(region.isEmpty() ? jsUndefined() : jsString(vm, region)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getTimezones
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetTimeZones, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getTimeZones called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->timeZones(globalObject)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getTextInfo
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetTextInfo, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getTextInfo called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->textInfo(globalObject)));
}

// https://tc39.es/proposal-intl-locale-info/#sec-Intl.Locale.prototype.getWeekInfo
JSC_DEFINE_HOST_FUNCTION(intlLocalePrototypeFuncGetWeekInfo, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* locale = jsDynamicCast<IntlLocale*>(callFrame->thisValue());
    if (UNLIKELY(!locale))
        return throwVMTypeError(globalObject, scope, "Intl.Locale.prototype.getWeekInfo called on value that's not a Locale"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(locale->weekInfo(globalObject)));
}

} // namespace JSC
