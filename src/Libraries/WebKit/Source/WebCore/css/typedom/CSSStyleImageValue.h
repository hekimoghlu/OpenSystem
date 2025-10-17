/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

#include "CSSImageValue.h"
#include "CSSStyleValue.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;

class CSSStyleImageValue final : public CSSStyleValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSStyleImageValue);
public:
    static Ref<CSSStyleImageValue> create(Ref<CSSImageValue>&& cssValue, Document* document)
    {
        return adoptRef(*new CSSStyleImageValue(WTFMove(cssValue), document));
    }

    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;

    CachedImage* image() { return m_cssValue->cachedImage(); }
    bool isLoadedFromOpaqueSource() const { return m_cssValue->isLoadedFromOpaqueSource(); }
    Document* document() const;
    
    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSStyleImageValue; }
    
    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSStyleImageValue(Ref<CSSImageValue>&&, Document*);

    Ref<CSSImageValue> m_cssValue;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSStyleImageValue)
    static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSStyleImageValue; }
SPECIALIZE_TYPE_TRAITS_END()
