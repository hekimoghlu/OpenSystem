/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "IntlSegmenter.h"

#include "IntlObjectInlines.h"
#include "IntlSegments.h"
#include "IntlWorkaround.h"
#include "JSCInlines.h"
#include "ObjectConstructor.h"

namespace JSC {

const ClassInfo IntlSegmenter::s_info = { "Object"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(IntlSegmenter) };

IntlSegmenter* IntlSegmenter::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlSegmenter>(vm)) IntlSegmenter(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlSegmenter::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlSegmenter::IntlSegmenter(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

// https://tc39.es/proposal-intl-segmenter/#sec-intl.segmenter
void IntlSegmenter::initializeSegmenter(JSGlobalObject* globalObject, JSValue locales, JSValue optionsValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto requestedLocales = canonicalizeLocaleList(globalObject, locales);
    RETURN_IF_EXCEPTION(scope, void());

    JSObject* options = intlGetOptionsObject(globalObject, optionsValue);
    RETURN_IF_EXCEPTION(scope, void());

    ResolveLocaleOptions localeOptions;

    LocaleMatcher localeMatcher = intlOption<LocaleMatcher>(globalObject, options, vm.propertyNames->localeMatcher, { { "lookup"_s, LocaleMatcher::Lookup }, { "best fit"_s, LocaleMatcher::BestFit } }, "localeMatcher must be either \"lookup\" or \"best fit\""_s, LocaleMatcher::BestFit);
    RETURN_IF_EXCEPTION(scope, void());

    auto localeData = [](const String&, RelevantExtensionKey) -> Vector<String> {
        return { };
    };

    const auto& availableLocales = intlSegmenterAvailableLocales();
    auto resolved = resolveLocale(globalObject, availableLocales, requestedLocales, localeMatcher, localeOptions, { }, localeData);

    m_locale = resolved.locale;
    if (m_locale.isEmpty()) {
        throwTypeError(globalObject, scope, "failed to initialize Segmenter due to invalid locale"_s);
        return;
    }

    m_granularity = intlOption<Granularity>(globalObject, options, vm.propertyNames->granularity, { { "grapheme"_s, Granularity::Grapheme }, { "word"_s, Granularity::Word }, { "sentence"_s, Granularity::Sentence } }, "granularity must be either \"grapheme\", \"word\", or \"sentence\""_s, Granularity::Grapheme);
    RETURN_IF_EXCEPTION(scope, void());

    UBreakIteratorType type = UBRK_CHARACTER;
    switch (m_granularity) {
    case Granularity::Grapheme:
        type = UBRK_CHARACTER;
        break;
    case Granularity::Word:
        type = UBRK_WORD;
        break;
    case Granularity::Sentence:
        type = UBRK_SENTENCE;
        break;
    }

    UErrorCode status = U_ZERO_ERROR;
    m_segmenter = std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>(ubrk_open(type, m_locale.utf8().data(), nullptr, 0, &status));
    if (U_FAILURE(status)) {
        throwTypeError(globalObject, scope, "failed to initialize Segmenter"_s);
        return;
    }
}

// https://tc39.es/proposal-intl-segmenter/#sec-intl.segmenter.prototype.segment
JSValue IntlSegmenter::segment(JSGlobalObject* globalObject, JSValue stringValue) const
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSString* jsString = stringValue.toString(globalObject);
    RETURN_IF_EXCEPTION(scope, { });
    auto string = jsString->value(globalObject);
    RETURN_IF_EXCEPTION(scope, { });
    auto expectedCharacters = string->charactersWithoutNullTermination();
    if (!expectedCharacters) {
        throwOutOfMemoryError(globalObject, scope);
        return { };
    }

    auto upconvertedCharacters = Box<Vector<UChar>>::create(expectedCharacters.value());

    UErrorCode status = U_ZERO_ERROR;
    auto segmenter = std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>(cloneUBreakIterator(m_segmenter.get(), &status));
    if (U_FAILURE(status)) {
        throwTypeError(globalObject, scope, "failed to initialize Segments"_s);
        return { };
    }
    ubrk_setText(segmenter.get(), upconvertedCharacters->data(), upconvertedCharacters->size(), &status);
    if (U_FAILURE(status)) {
        throwTypeError(globalObject, scope, "failed to initialize Segments"_s);
        return { };
    }

    return IntlSegments::create(vm, globalObject->segmentsStructure(), WTFMove(segmenter), WTFMove(upconvertedCharacters), jsString, m_granularity);
}

// https://tc39.es/proposal-intl-segmenter/#sec-intl.segmenter.prototype.resolvedoptions
JSObject* IntlSegmenter::resolvedOptions(JSGlobalObject* globalObject) const
{
    VM& vm = globalObject->vm();
    JSObject* options = constructEmptyObject(globalObject);
    options->putDirect(vm, vm.propertyNames->locale, jsString(vm, m_locale));
    options->putDirect(vm, vm.propertyNames->granularity, jsNontrivialString(vm, granularityString(m_granularity)));
    return options;
}

ASCIILiteral IntlSegmenter::granularityString(Granularity granularity)
{
    switch (granularity) {
    case Granularity::Grapheme:
        return "grapheme"_s;
    case Granularity::Word:
        return "word"_s;
    case Granularity::Sentence:
        return "sentence"_s;
    }
    ASSERT_NOT_REACHED();
    return { };
}

JSObject* IntlSegmenter::createSegmentDataObject(JSGlobalObject* globalObject, JSString* string, int32_t startIndex, int32_t endIndex, UBreakIterator& segmenter, Granularity granularity)
{
    VM& vm = globalObject->vm();
    JSObject* result = constructEmptyObject(globalObject);
    result->putDirect(vm, vm.propertyNames->segment, jsSubstring(globalObject, string, startIndex, endIndex - startIndex));
    result->putDirect(vm, vm.propertyNames->index, jsNumber(startIndex));
    result->putDirect(vm, vm.propertyNames->input, string);
    if (granularity == IntlSegmenter::Granularity::Word) {
        int32_t ruleStatus = ubrk_getRuleStatus(&segmenter);
        result->putDirect(vm, vm.propertyNames->isWordLike, jsBoolean(!(ruleStatus >= UBRK_WORD_NONE && ruleStatus <= UBRK_WORD_NONE_LIMIT)));
    }
    return result;
}

} // namespace JSC
