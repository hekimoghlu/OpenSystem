/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#include "QualifiedName.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Element;
class WeakPtrImplWithEventTargetData;

class CustomElementDefaultARIA final : public CanMakeCheckedPtr<CustomElementDefaultARIA> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CustomElementDefaultARIA);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CustomElementDefaultARIA);
public:
    CustomElementDefaultARIA();
    ~CustomElementDefaultARIA();

    bool hasAttribute(const QualifiedName&) const;
    const AtomString& valueForAttribute(const Element& thisElement, const QualifiedName&) const;
    void setValueForAttribute(const QualifiedName&, const AtomString&);
    RefPtr<Element> elementForAttribute(const Element& thisElement, const QualifiedName&) const;
    void setElementForAttribute(const QualifiedName&, Element*);
    Vector<Ref<Element>> elementsForAttribute(const Element& thisElement, const QualifiedName&) const;
    void setElementsForAttribute(const QualifiedName&, std::optional<Vector<Ref<Element>>>&&);

private:
    using WeakElementPtr = WeakPtr<Element, WeakPtrImplWithEventTargetData>;
    UncheckedKeyHashMap<QualifiedName, std::variant<AtomString, WeakElementPtr, Vector<WeakElementPtr>>> m_map;
};

}; // namespace WebCore
