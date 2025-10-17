/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class CSSCustomPropertyValue;
class CSSVariableData;
class Document;
class WeakPtrImplWithEventTargetData;

enum class ConstantProperty {
    SafeAreaInsetTop,
    SafeAreaInsetRight,
    SafeAreaInsetBottom,
    SafeAreaInsetLeft,
    FullscreenInsetTop,
    FullscreenInsetRight,
    FullscreenInsetBottom,
    FullscreenInsetLeft,
    FullscreenAutoHideDuration,
};

class ConstantPropertyMap {
    WTF_MAKE_TZONE_ALLOCATED(ConstantPropertyMap);
public:
    explicit ConstantPropertyMap(Document&);

    typedef UncheckedKeyHashMap<AtomString, Ref<CSSCustomPropertyValue>> Values;
    const Values& values() const;

    void didChangeSafeAreaInsets();
    void didChangeFullscreenInsets();
    void setFullscreenAutoHideDuration(Seconds);

private:
    void buildValues();

    const AtomString& nameForProperty(ConstantProperty) const;
    void setValueForProperty(ConstantProperty, Ref<CSSVariableData>&&);

    void updateConstantsForSafeAreaInsets();
    void updateConstantsForFullscreen();

    Ref<Document> protectedDocument() const;

    std::optional<Values> m_values;

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
