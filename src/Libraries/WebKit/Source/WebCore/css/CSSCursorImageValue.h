/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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

#include "CSSValue.h"
#include "CSSValuePair.h"
#include "IntPoint.h"
#include "ResourceLoaderOptions.h"
#include <wtf/WeakHashSet.h>

namespace WebCore {

class StyleCursorImage;
class StyleImage;

namespace Style {
class BuilderState;
}

class CSSCursorImageValue final : public CSSValue {
public:
    static Ref<CSSCursorImageValue> create(Ref<CSSValue>&& imageValue, RefPtr<CSSValue>&& hotSpot, LoadedFromOpaqueSource);
    static Ref<CSSCursorImageValue> create(Ref<CSSValue>&& imageValue, RefPtr<CSSValue>&& hotSpot, URL, LoadedFromOpaqueSource);
    ~CSSCursorImageValue();

    const URL& imageURL() const { return m_originalURL; }
    String customCSSText() const;
    bool equals(const CSSCursorImageValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        return func(m_imageValue.get());
    }

    RefPtr<StyleCursorImage> createStyleImage(const Style::BuilderState&) const;

private:
    CSSCursorImageValue(Ref<CSSValue>&& imageValue, RefPtr<CSSValue>&& hotSpot, URL, LoadedFromOpaqueSource);

    URL m_originalURL;
    Ref<CSSValue> m_imageValue;
    RefPtr<CSSValue> m_hotSpot;
    LoadedFromOpaqueSource m_loadedFromOpaqueSource { LoadedFromOpaqueSource::No };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSCursorImageValue, isCursorImageValue())
