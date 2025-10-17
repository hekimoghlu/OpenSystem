/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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

#include "CSSPrimitiveValue.h"
#include "CSSValue.h"
#include <wtf/Function.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSImageSetOptionValue final : public CSSValue {
public:
    static Ref<CSSImageSetOptionValue> create(Ref<CSSValue>&&);
    static Ref<CSSImageSetOptionValue> create(Ref<CSSValue>&&, Ref<CSSPrimitiveValue>&&);
    static Ref<CSSImageSetOptionValue> create(Ref<CSSValue>&&, Ref<CSSPrimitiveValue>&&, String);

    bool equals(const CSSImageSetOptionValue&) const;
    String customCSSText() const;

    Ref<CSSValue> image() const { return m_image; }

    Ref<CSSPrimitiveValue> resolution() const { return m_resolution; }
    void setResolution(Ref<CSSPrimitiveValue>&&);

    String type() const { return m_mimeType; }
    void setType(String);

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_image.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_resolution.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }
    bool customTraverseSubresources(const Function<bool(const CachedResource&)>&) const;
    void customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>&);
    void customClearReplacementURLForSubresources();

private:
    CSSImageSetOptionValue(Ref<CSSValue>&&, Ref<CSSPrimitiveValue>&&);
    CSSImageSetOptionValue(Ref<CSSValue>&&, Ref<CSSPrimitiveValue>&&, String&&);

    Ref<CSSValue> m_image;
    Ref<CSSPrimitiveValue> m_resolution;
    String m_mimeType;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSImageSetOptionValue, isImageSetOptionValue())
