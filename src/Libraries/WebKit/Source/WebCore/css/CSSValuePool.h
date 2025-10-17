/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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

#include "CSSColorValue.h"
#include "CSSPrimitiveValue.h"
#include "ColorHash.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class CSSValueList;
class CSSValuePool;

class StaticCSSValuePool {
    friend class CSSPrimitiveValue;
    friend class CSSValuePool;
    friend class LazyNeverDestroyed<StaticCSSValuePool>;

public:
    static void init();

private:
    StaticCSSValuePool();

    LazyNeverDestroyed<CSSPrimitiveValue> m_implicitInitialValue;

    LazyNeverDestroyed<CSSColorValue> m_transparentColor;
    LazyNeverDestroyed<CSSColorValue> m_whiteColor;
    LazyNeverDestroyed<CSSColorValue> m_blackColor;

    static constexpr int maximumCacheableIntegerValue = 255;

    std::array<LazyNeverDestroyed<CSSPrimitiveValue>, maximumCacheableIntegerValue + 1> m_pixelValues;
    std::array<LazyNeverDestroyed<CSSPrimitiveValue>, maximumCacheableIntegerValue + 1> m_percentageValues;
    std::array<LazyNeverDestroyed<CSSPrimitiveValue>, maximumCacheableIntegerValue + 1> m_numberValues;
    std::array<LazyNeverDestroyed<CSSPrimitiveValue>, numCSSValueKeywords> m_identifierValues;
};

WEBCORE_EXPORT extern LazyNeverDestroyed<StaticCSSValuePool> staticCSSValuePool;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSValuePool);
class CSSValuePool {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CSSValuePool);
    WTF_MAKE_NONCOPYABLE(CSSValuePool);
public:
    CSSValuePool();
    static CSSValuePool& singleton();
    void drain();

    Ref<CSSColorValue> createColorValue(const WebCore::Color&);
    RefPtr<CSSValueList> createFontFaceValue(const AtomString&);
    Ref<CSSPrimitiveValue> createFontFamilyValue(const AtomString&);

private:
    UncheckedKeyHashMap<WebCore::Color, Ref<CSSColorValue>> m_colorValueCache;
    UncheckedKeyHashMap<AtomString, RefPtr<CSSValueList>> m_fontFaceValueCache;
    UncheckedKeyHashMap<AtomString, Ref<CSSPrimitiveValue>> m_fontFamilyValueCache;
};

inline CSSPrimitiveValue& CSSPrimitiveValue::implicitInitialValue()
{
    return staticCSSValuePool->m_implicitInitialValue.get();
}

inline Ref<CSSPrimitiveValue> CSSPrimitiveValue::create(CSSValueID identifier)
{
    RELEASE_ASSERT(identifier < numCSSValueKeywords);
    return staticCSSValuePool->m_identifierValues[identifier].get();
}

} // namespace WebCore
