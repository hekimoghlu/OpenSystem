/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
#include "FontFace.h"

#include "CSSFontFaceSource.h"
#include "CSSFontSelector.h"
#include "CSSPrimitiveValueMappings.h"
#include "CSSPropertyParserConsumer+Font.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"
#include "DOMPromiseProxy.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "JSFontFace.h"
#include "TrustedFonts.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/ArrayBufferView.h>
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {

static bool populateFontFaceWithArrayBuffer(CSSFontFace& fontFace, Ref<JSC::ArrayBufferView>&& arrayBufferView)
{
    auto source = makeUnique<CSSFontFaceSource>(fontFace, WTFMove(arrayBufferView));
    fontFace.adoptSource(WTFMove(source));
    return false;
}

void FontFace::setErrorState()
{
    m_loadedPromise->reject(Exception { ExceptionCode::SyntaxError });
    m_backing->setErrorState();
}

Ref<FontFace> FontFace::create(ScriptExecutionContext& context, const String& family, Source&& source, const Descriptors& descriptors)
{
    ASSERT(context.cssFontSelector());
    auto result = adoptRef(*new FontFace(*context.cssFontSelector()));
    result->suspendIfNeeded();

#if COMPILER(GCC) && (CPU(ARM) || CPU(ARM64))
    // FIXME: Workaround for GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115033
    // that is related to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115135 as well.
    volatile
#endif
    bool dataRequiresAsynchronousLoading = true;

    auto setFamilyResult = result->setFamily(context, family);
    if (setFamilyResult.hasException()) {
        result->setErrorState();
        return result;
    }

    auto fontTrustedTypes = context.settingsValues().downloadableBinaryFontTrustedTypes;
    auto sourceConversionResult = WTF::switchOn(source,
        [&] (String& string) -> ExceptionOr<void> {
            auto* document = dynamicDowncast<Document>(context);
            auto value = CSSPropertyParserHelpers::parseFontFaceSrc(string, document ? CSSParserContext(*document) : HTMLStandardMode);
            if (!value)
                return Exception { ExceptionCode::SyntaxError };
            CSSFontFace::appendSources(result->backing(), *value, &context, false);
            return { };
        },
        [&, fontTrustedTypes] (RefPtr<ArrayBufferView>& arrayBufferView) -> ExceptionOr<void> {
            if (!arrayBufferView || fontBinaryParsingPolicy(arrayBufferView->span(), fontTrustedTypes) == FontParsingPolicy::Deny)
                return { };

            dataRequiresAsynchronousLoading = populateFontFaceWithArrayBuffer(result->backing(), arrayBufferView.releaseNonNull());
            return { };
        },
        [&, fontTrustedTypes] (RefPtr<ArrayBuffer>& arrayBuffer) -> ExceptionOr<void> {
            if (!arrayBuffer || fontBinaryParsingPolicy(arrayBuffer->span(), fontTrustedTypes) == FontParsingPolicy::Deny)
                return { };

            unsigned byteLength = arrayBuffer->byteLength();
            auto arrayBufferView = JSC::Uint8Array::create(WTFMove(arrayBuffer), 0, byteLength);
            dataRequiresAsynchronousLoading = populateFontFaceWithArrayBuffer(result->backing(), WTFMove(arrayBufferView));
            return { };
        }
    );

    if (sourceConversionResult.hasException()) {
        result->setErrorState();
        return result;
    }

    // These ternaries match the default strings inside the FontFaceDescriptors dictionary inside FontFace.idl.
    auto setStyleResult = result->setStyle(context, descriptors.style.isEmpty() ? "normal"_s : descriptors.style);
    if (setStyleResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setWeightResult = result->setWeight(context, descriptors.weight.isEmpty() ? "normal"_s : descriptors.weight);
    if (setWeightResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setWidthResult = result->setWidth(context, descriptors.width.isEmpty() ? "normal"_s : descriptors.width);
    if (setWidthResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setUnicodeRangeResult = result->setUnicodeRange(context, descriptors.unicodeRange.isEmpty() ? "U+0-10FFFF"_s : descriptors.unicodeRange);
    if (setUnicodeRangeResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setFeatureSettingsResult = result->setFeatureSettings(context, descriptors.featureSettings.isEmpty() ? "normal"_s : descriptors.featureSettings);
    if (setFeatureSettingsResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setDisplayResult = result->setDisplay(context, descriptors.display.isEmpty() ? "auto"_s : descriptors.display);
    if (setDisplayResult.hasException()) {
        result->setErrorState();
        return result;
    }
    auto setSizeAdjustResult = result->setSizeAdjust(context, descriptors.sizeAdjust.isEmpty() ? "100%"_s : descriptors.sizeAdjust);
    if (setSizeAdjustResult.hasException()) {
        result->setErrorState();
        return result;
    }

    if (!dataRequiresAsynchronousLoading) {
        result->backing().load();
        auto status = result->backing().status();
        ASSERT_UNUSED(status, status == CSSFontFace::Status::Success || status == CSSFontFace::Status::Failure);
    }

    return result;
}

Ref<FontFace> FontFace::create(ScriptExecutionContext* context, CSSFontFace& face)
{
    auto fontFace = adoptRef(*new FontFace(context, face));
    fontFace->suspendIfNeeded();
    return fontFace;
}

FontFace::FontFace(CSSFontSelector& fontSelector)
    : ActiveDOMObject(fontSelector.scriptExecutionContext())
    , m_backing(CSSFontFace::create(fontSelector, nullptr, this))
    , m_loadedPromise(makeUniqueRef<LoadedPromise>(*this, &FontFace::loadedPromiseResolve))
{
    m_backing->addClient(*this);
}

FontFace::FontFace(ScriptExecutionContext* context, CSSFontFace& face)
    : ActiveDOMObject(context)
    , m_backing(face)
    , m_loadedPromise(makeUniqueRef<LoadedPromise>(*this, &FontFace::loadedPromiseResolve))
{
    m_backing->addClient(*this);
}

FontFace::~FontFace()
{
    m_backing->removeClient(*this);
}

ExceptionOr<void> FontFace::setFamily(ScriptExecutionContext& context, const String& family)
{
    if (family.isNull())
        return Exception { ExceptionCode::SyntaxError };
    // FIXME: Don't use a list here. https://bugs.webkit.org/show_bug.cgi?id=196381
    m_backing->setFamilies(CSSValueList::createCommaSeparated(context.cssValuePool().createFontFamilyValue(AtomString { family })));
    return { };
}

ExceptionOr<void> FontFace::setStyle(ScriptExecutionContext& context, const String& style)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceFontStyle(style, context)) {
        m_backing->setStyle(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setWeight(ScriptExecutionContext& context, const String& weight)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceFontWeight(weight, context)) {
        m_backing->setWeight(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setWidth(ScriptExecutionContext& context, const String& width)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceFontWidth(width, context)) {
        m_backing->setWidth(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setUnicodeRange(ScriptExecutionContext& context, const String& unicodeRange)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceUnicodeRange(unicodeRange, context)) {
        m_backing->setUnicodeRange(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setFeatureSettings(ScriptExecutionContext& context, const String& featureSettings)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceFeatureSettings(featureSettings, context)) {
        m_backing->setFeatureSettings(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setDisplay(ScriptExecutionContext& context, const String& display)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceDisplay(display, context)) {
        m_backing->setDisplay(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

ExceptionOr<void> FontFace::setSizeAdjust(ScriptExecutionContext& context, const String& sizeAdjust)
{
    if (auto value = CSSPropertyParserHelpers::parseFontFaceSizeAdjust(sizeAdjust, context)) {
        m_backing->setSizeAdjust(*value);
        return { };
    }
    return Exception { ExceptionCode::SyntaxError };
}

String FontFace::family() const
{
    if (auto value = m_backing->family(); !value.isNull())
        return value;
    return "normal"_s;
}

String FontFace::style() const
{
    if (auto value = m_backing->style(); !value.isNull())
        return value;
    return "normal"_s;
}

String FontFace::weight() const
{
    if (auto value = m_backing->weight(); !value.isNull())
        return value;
    return "normal"_s;
}

String FontFace::width() const
{
    if (auto value = m_backing->width(); !value.isNull())
        return value;
    return "normal"_s;
}

String FontFace::unicodeRange() const
{
    if (auto value = m_backing->unicodeRange(); !value.isNull())
        return value;
    return "U+0-10FFFF"_s;
}

String FontFace::featureSettings() const
{
    if (auto value = m_backing->featureSettings(); !value.isNull())
        return value;
    return "normal"_s;
}

String FontFace::sizeAdjust() const
{
    if (auto value = m_backing->sizeAdjust(); !value.isNull())
        return value;
    return "100%"_s;
}

String FontFace::display() const
{
    if (auto value = m_backing->display(); !value.isNull())
        return value;
    return autoAtom();
}

auto FontFace::status() const -> LoadStatus
{
    switch (m_backing->status()) {
    case CSSFontFace::Status::Pending:
        return LoadStatus::Unloaded;
    case CSSFontFace::Status::Loading:
        return LoadStatus::Loading;
    case CSSFontFace::Status::TimedOut:
        return LoadStatus::Error;
    case CSSFontFace::Status::Success:
        return LoadStatus::Loaded;
    case CSSFontFace::Status::Failure:
        return LoadStatus::Error;
    }
    ASSERT_NOT_REACHED();
    return LoadStatus::Error;
}

void FontFace::adopt(CSSFontFace& newFace)
{
    m_backing->removeClient(*this);
    m_backing = newFace;
    m_backing->addClient(*this);
    newFace.setWrapper(*this);
}

void FontFace::fontStateChanged(CSSFontFace& face, CSSFontFace::Status, CSSFontFace::Status newState)
{
    ASSERT_UNUSED(face, &face == m_backing.ptr());
    switch (newState) {
    case CSSFontFace::Status::Loading:
        break;
    case CSSFontFace::Status::TimedOut:
        break;
    case CSSFontFace::Status::Success:
        // FIXME: This check should not be needed, but because FontFace's are sometimes adopted after they have already
        // gone through a load cycle, we can sometimes come back through here and try to resolve the promise again.
        if (!m_loadedPromise->isFulfilled())
            m_loadedPromise->resolve(*this);
        return;
    case CSSFontFace::Status::Failure:
        // FIXME: This check should not be needed, but because FontFace's are sometimes adopted after they have already
        // gone through a load cycle, we can sometimes come back through here and try to resolve the promise again.
        if (!m_loadedPromise->isFulfilled())
            m_loadedPromise->reject(Exception { ExceptionCode::NetworkError });
        return;
    case CSSFontFace::Status::Pending:
        ASSERT_NOT_REACHED();
        return;
    }
}

auto FontFace::loadForBindings() -> LoadedPromise&
{
    m_mayLoadedPromiseBeScriptObservable = true;
    m_backing->load();
    return m_loadedPromise.get();
}

auto FontFace::loadedForBindings() -> LoadedPromise&
{
    m_mayLoadedPromiseBeScriptObservable = true;
    return m_loadedPromise.get();
}

FontFace& FontFace::loadedPromiseResolve()
{
    return *this;
}

bool FontFace::virtualHasPendingActivity() const
{
    return m_mayLoadedPromiseBeScriptObservable && !m_loadedPromise->isFulfilled();
}

}
