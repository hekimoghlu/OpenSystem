/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/WeakRef.h>

namespace WebCore {

class Attr;

class NamedNodeMap final : public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NamedNodeMap);
public:
    explicit NamedNodeMap(Element& element)
        : m_element(element)
    {
    }

    WEBCORE_EXPORT void ref();
    WEBCORE_EXPORT void deref();

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    WEBCORE_EXPORT unsigned length() const;
    WEBCORE_EXPORT RefPtr<Attr> item(unsigned index) const;
    WEBCORE_EXPORT RefPtr<Attr> getNamedItem(const AtomString&) const;
    WEBCORE_EXPORT RefPtr<Attr> getNamedItemNS(const AtomString& namespaceURI, const AtomString& localName) const;
    WEBCORE_EXPORT ExceptionOr<RefPtr<Attr>> setNamedItem(Attr&);
    WEBCORE_EXPORT ExceptionOr<Ref<Attr>> removeNamedItem(const AtomString& name);
    WEBCORE_EXPORT ExceptionOr<Ref<Attr>> removeNamedItemNS(const AtomString& namespaceURI, const AtomString& localName);
    bool isSupportedPropertyName(const AtomString&) const;

    Vector<String> supportedPropertyNames() const;

    Element& element();
    Ref<Element> protectedElement() const;

private:
    WeakRef<Element, WeakPtrImplWithEventTargetData> m_element;
};

} // namespace WebCore
