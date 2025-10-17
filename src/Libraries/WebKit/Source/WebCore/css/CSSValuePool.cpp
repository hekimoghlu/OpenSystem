/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "CSSValuePool.h"

#include "CSSParser.h"
#include "CSSPrimitiveValueMappings.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSValuePool);

LazyNeverDestroyed<StaticCSSValuePool> staticCSSValuePool;

StaticCSSValuePool::StaticCSSValuePool()
{
    m_implicitInitialValue.construct(CSSValue::StaticCSSValue, CSSPrimitiveValue::ImplicitInitialValue);
    
    m_transparentColor.construct(CSSValue::StaticCSSValue, WebCore::Color::transparentBlack);
    m_whiteColor.construct(CSSValue::StaticCSSValue, WebCore::Color::white);
    m_blackColor.construct(CSSValue::StaticCSSValue, WebCore::Color::black);

    for (auto keyword : allCSSValueKeywords())
        m_identifierValues[enumToUnderlyingType(keyword)].construct(CSSValue::StaticCSSValue, keyword);

    for (unsigned i = 0; i <= maximumCacheableIntegerValue; ++i) {
        m_pixelValues[i].construct(CSSValue::StaticCSSValue, i, CSSUnitType::CSS_PX);
        m_percentageValues[i].construct(CSSValue::StaticCSSValue, i, CSSUnitType::CSS_PERCENTAGE);
        m_numberValues[i].construct(CSSValue::StaticCSSValue, i, CSSUnitType::CSS_NUMBER);
    }
}

void StaticCSSValuePool::init()
{
    static std::once_flag onceKey;
    std::call_once(onceKey, []() {
        staticCSSValuePool.construct();
    });
}

CSSValuePool::CSSValuePool() = default;

// FIXME: This function needs a name that make it clear that this is not the one and only CSSValuePool, rather the one that can be used only on the main thread.
// FIXME: Consider a design where the value pool thread-local storage so callers don't have to deal directly with the value pool at all.
CSSValuePool& CSSValuePool::singleton()
{
    static MainThreadNeverDestroyed<CSSValuePool> pool;
    return pool;
}

Ref<CSSColorValue> CSSValuePool::createColorValue(const WebCore::Color& color)
{
    // These are the empty and deleted values of the hash table.
    if (color == WebCore::Color::transparentBlack)
        return staticCSSValuePool->m_transparentColor.get();
    if (color == WebCore::Color::white)
        return staticCSSValuePool->m_whiteColor.get();
    // Just because it is common.
    if (color == WebCore::Color::black)
        return staticCSSValuePool->m_blackColor.get();

    // Remove one entry at random if the cache grows too large.
    // FIXME: Use TinyLRUCache instead?
    const int maximumColorCacheSize = 512;
    if (m_colorValueCache.size() >= maximumColorCacheSize)
        m_colorValueCache.remove(m_colorValueCache.random());

    return m_colorValueCache.ensure(color, [&color] {
        return CSSColorValue::create(color);
    }).iterator->value;
}

Ref<CSSPrimitiveValue> CSSValuePool::createFontFamilyValue(const AtomString& familyName)
{
    // Remove one entry at random if the cache grows too large.
    // FIXME: Use TinyLRUCache instead?
    const int maximumFontFamilyCacheSize = 128;
    if (m_fontFamilyValueCache.size() >= maximumFontFamilyCacheSize)
        m_fontFamilyValueCache.remove(m_fontFamilyValueCache.random());

    return m_fontFamilyValueCache.ensure(familyName, [&familyName] {
        return CSSPrimitiveValue::createFontFamily(familyName);
    }).iterator->value;
}

RefPtr<CSSValueList> CSSValuePool::createFontFaceValue(const AtomString& string)
{
    // Remove one entry at random if the cache grows too large.
    // FIXME: Use TinyLRUCache instead?
    const int maximumFontFaceCacheSize = 128;
    if (m_fontFaceValueCache.size() >= maximumFontFaceCacheSize)
        m_fontFaceValueCache.remove(m_fontFaceValueCache.random());

    return m_fontFaceValueCache.ensure(string, [&string]() -> RefPtr<CSSValueList> {
        auto value = CSSParser::parseSingleValue(CSSPropertyFontFamily, string);
        return dynamicDowncast<CSSValueList>(value.get());
    }).iterator->value;
}

void CSSValuePool::drain()
{
    m_colorValueCache.clear();
    m_fontFaceValueCache.clear();
    m_fontFamilyValueCache.clear();
}

}
