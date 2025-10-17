/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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

#include "Element.h"
#include "JSDOMSetLike.h"
#include "ScriptWrappable.h"

namespace WebCore {

class CustomStateSet final : public ScriptWrappable, public RefCounted<CustomStateSet> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CustomStateSet);

public:
    static Ref<CustomStateSet> create(Element& element)
    {
        return adoptRef(*new CustomStateSet(element));
    };

    bool addToSetLike(const AtomString& state);
    bool removeFromSetLike(const AtomString& state);
    void clearFromSetLike();
    void initializeSetLike(DOMSetAdapter&) { };

    bool has(const AtomString&) const;

private:
    explicit CustomStateSet(Element& element)
        : m_element(element)
    {
    }

    ListHashSet<AtomString> m_states;

    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_element;
};

} // namespace WebCore
