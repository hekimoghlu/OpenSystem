/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#include "ConstantPropertyMap.h"

#include "CSSCustomPropertyValue.h"
#include "CSSParserTokenRange.h"
#include "CSSVariableData.h"
#include "Document.h"
#include "Page.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ConstantPropertyMap);

ConstantPropertyMap::ConstantPropertyMap(Document& document)
    : m_document(document)
{
}

const ConstantPropertyMap::Values& ConstantPropertyMap::values() const
{
    if (!m_values)
        const_cast<ConstantPropertyMap&>(*this).buildValues();
    return *m_values;
}

const AtomString& ConstantPropertyMap::nameForProperty(ConstantProperty property) const
{
    static MainThreadNeverDestroyed<const AtomString> safeAreaInsetTopName("safe-area-inset-top"_s);
    static MainThreadNeverDestroyed<const AtomString> safeAreaInsetRightName("safe-area-inset-right"_s);
    static MainThreadNeverDestroyed<const AtomString> safeAreaInsetBottomName("safe-area-inset-bottom"_s);
    static MainThreadNeverDestroyed<const AtomString> safeAreaInsetLeftName("safe-area-inset-left"_s);
    static MainThreadNeverDestroyed<const AtomString> fullscreenInsetTopName("fullscreen-inset-top"_s);
    static MainThreadNeverDestroyed<const AtomString> fullscreenInsetLeftName("fullscreen-inset-left"_s);
    static MainThreadNeverDestroyed<const AtomString> fullscreenInsetBottomName("fullscreen-inset-bottom"_s);
    static MainThreadNeverDestroyed<const AtomString> fullscreenInsetRightName("fullscreen-inset-right"_s);
    static MainThreadNeverDestroyed<const AtomString> fullscreenAutoHideDurationName("fullscreen-auto-hide-duration"_s);

    switch (property) {
    case ConstantProperty::SafeAreaInsetTop:
        return safeAreaInsetTopName;
    case ConstantProperty::SafeAreaInsetRight:
        return safeAreaInsetRightName;
    case ConstantProperty::SafeAreaInsetBottom:
        return safeAreaInsetBottomName;
    case ConstantProperty::SafeAreaInsetLeft:
        return safeAreaInsetLeftName;
    case ConstantProperty::FullscreenInsetTop:
        return fullscreenInsetTopName;
    case ConstantProperty::FullscreenInsetLeft:
        return fullscreenInsetLeftName;
    case ConstantProperty::FullscreenInsetBottom:
        return fullscreenInsetBottomName;
    case ConstantProperty::FullscreenInsetRight:
        return fullscreenInsetRightName;
    case ConstantProperty::FullscreenAutoHideDuration:
        return fullscreenAutoHideDurationName;
    }

    return nullAtom();
}

void ConstantPropertyMap::setValueForProperty(ConstantProperty property, Ref<CSSVariableData>&& data)
{
    if (!m_values)
        buildValues();

    auto& name = nameForProperty(property);
    m_values->set(name, CSSCustomPropertyValue::createSyntaxAll(name, WTFMove(data)));
}

void ConstantPropertyMap::buildValues()
{
    m_values = Values { };

    updateConstantsForSafeAreaInsets();
    updateConstantsForFullscreen();
}

static Ref<CSSVariableData> variableDataForPositivePixelLength(float lengthInPx)
{
    ASSERT(lengthInPx >= 0);

    CSSParserToken token(lengthInPx, NumberValueType, NoSign, { });
    token.convertToDimensionWithUnit("px"_s);

    Vector<CSSParserToken> tokens { token };
    CSSParserTokenRange tokenRange(tokens);
    return CSSVariableData::create(tokenRange);
}

static Ref<CSSVariableData> variableDataForPositiveDuration(Seconds durationInSeconds)
{
    ASSERT(durationInSeconds >= 0_s);

    CSSParserToken token(durationInSeconds.value(), NumberValueType, NoSign, { });
    token.convertToDimensionWithUnit("s"_s);

    Vector<CSSParserToken> tokens { token };
    CSSParserTokenRange tokenRange(tokens);
    return CSSVariableData::create(tokenRange);
}

Ref<Document> ConstantPropertyMap::protectedDocument() const
{
    return m_document.get();
}

void ConstantPropertyMap::updateConstantsForSafeAreaInsets()
{
    RefPtr page = m_document->page();
    FloatBoxExtent unobscuredSafeAreaInsets = page ? page->unobscuredSafeAreaInsets() : FloatBoxExtent();
    setValueForProperty(ConstantProperty::SafeAreaInsetTop, variableDataForPositivePixelLength(unobscuredSafeAreaInsets.top()));
    setValueForProperty(ConstantProperty::SafeAreaInsetRight, variableDataForPositivePixelLength(unobscuredSafeAreaInsets.right()));
    setValueForProperty(ConstantProperty::SafeAreaInsetBottom, variableDataForPositivePixelLength(unobscuredSafeAreaInsets.bottom()));
    setValueForProperty(ConstantProperty::SafeAreaInsetLeft, variableDataForPositivePixelLength(unobscuredSafeAreaInsets.left()));
}

void ConstantPropertyMap::didChangeSafeAreaInsets()
{
    updateConstantsForSafeAreaInsets();
    protectedDocument()->invalidateMatchedPropertiesCacheAndForceStyleRecalc();
}

void ConstantPropertyMap::updateConstantsForFullscreen()
{
    RefPtr page = m_document->page();
    FloatBoxExtent fullscreenInsets = page ? page->fullscreenInsets() : FloatBoxExtent();
    setValueForProperty(ConstantProperty::FullscreenInsetTop, variableDataForPositivePixelLength(fullscreenInsets.top()));
    setValueForProperty(ConstantProperty::FullscreenInsetRight, variableDataForPositivePixelLength(fullscreenInsets.right()));
    setValueForProperty(ConstantProperty::FullscreenInsetBottom, variableDataForPositivePixelLength(fullscreenInsets.bottom()));
    setValueForProperty(ConstantProperty::FullscreenInsetLeft, variableDataForPositivePixelLength(fullscreenInsets.left()));

    Seconds fullscreenAutoHideDuration = page ? page->fullscreenAutoHideDuration() : 0_s;
    setValueForProperty(ConstantProperty::FullscreenAutoHideDuration, variableDataForPositiveDuration(fullscreenAutoHideDuration));
}

void ConstantPropertyMap::didChangeFullscreenInsets()
{
    updateConstantsForFullscreen();
    protectedDocument()->invalidateMatchedPropertiesCacheAndForceStyleRecalc();
}

void ConstantPropertyMap::setFullscreenAutoHideDuration(Seconds duration)
{
    setValueForProperty(ConstantProperty::FullscreenAutoHideDuration, variableDataForPositiveDuration(duration));
    protectedDocument()->invalidateMatchedPropertiesCacheAndForceStyleRecalc();
}

}
