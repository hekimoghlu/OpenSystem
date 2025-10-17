/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "CSSImageSetOptionValue.h"

#include "CSSImageValue.h"

namespace WebCore {

CSSImageSetOptionValue::CSSImageSetOptionValue(Ref<CSSValue>&& image, Ref<CSSPrimitiveValue>&& resolution)
    : CSSValue(ClassType::ImageSetOption)
    , m_image(WTFMove(image))
    , m_resolution(WTFMove(resolution))
{
}

CSSImageSetOptionValue::CSSImageSetOptionValue(Ref<CSSValue>&& image, Ref<CSSPrimitiveValue>&& resolution, String&& type)
    : CSSValue(ClassType::ImageSetOption)
    , m_image(WTFMove(image))
    , m_resolution(WTFMove(resolution))
    , m_mimeType(WTFMove(type))
{
}

Ref<CSSImageSetOptionValue> CSSImageSetOptionValue::create(Ref<CSSValue>&& image)
{
    ASSERT(is<CSSImageValue>(image) || image->isImageGeneratorValue());
    return adoptRef(*new CSSImageSetOptionValue(WTFMove(image), CSSPrimitiveValue::create(1.0, CSSUnitType::CSS_X)));
}

Ref<CSSImageSetOptionValue> CSSImageSetOptionValue::create(Ref<CSSValue>&& image, Ref<CSSPrimitiveValue>&& resolution)
{
    ASSERT(is<CSSImageValue>(image) || image->isImageGeneratorValue());
    return adoptRef(*new CSSImageSetOptionValue(WTFMove(image), WTFMove(resolution)));
}

Ref<CSSImageSetOptionValue> CSSImageSetOptionValue::create(Ref<CSSValue>&& image, Ref<CSSPrimitiveValue>&& resolution, String type)
{
    ASSERT(is<CSSImageValue>(image) || image->isImageGeneratorValue());
    return adoptRef(*new CSSImageSetOptionValue(WTFMove(image), WTFMove(resolution), WTFMove(type)));
}

bool CSSImageSetOptionValue::equals(const CSSImageSetOptionValue& other) const
{
    if (!m_image->equals(other.m_image))
        return false;

    if (!m_resolution->equals(other.m_resolution))
        return false;

    if (m_mimeType != other.m_mimeType)
        return false;

    return true;
}

String CSSImageSetOptionValue::customCSSText() const
{
    StringBuilder result;
    result.append(m_image->cssText());
    result.append(' ', m_resolution->cssText());
    if (!m_mimeType.isNull())
        result.append(" type(\""_s, m_mimeType, "\")"_s);

    return result.toString();
}

void CSSImageSetOptionValue::setResolution(Ref<CSSPrimitiveValue>&& resolution)
{
    m_resolution = WTFMove(resolution);
}

void CSSImageSetOptionValue::setType(String type)
{
    m_mimeType = WTFMove(type);
}

bool CSSImageSetOptionValue::customTraverseSubresources(const Function<bool(const CachedResource&)>& handler) const
{
    return m_resolution->traverseSubresources(handler) || m_image->traverseSubresources(handler);
}

void CSSImageSetOptionValue::customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>& replacementURLStrings)
{
    m_image->setReplacementURLForSubresources(replacementURLStrings);
    m_resolution->setReplacementURLForSubresources(replacementURLStrings);
}

void CSSImageSetOptionValue::customClearReplacementURLForSubresources()
{
    m_image->clearReplacementURLForSubresources();
    m_resolution->clearReplacementURLForSubresources();
}

} // namespace WebCore
