/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/WeakRef.h>

namespace WebCore {

class Element;
class WeakPtrImplWithEventTargetData;

class DatasetDOMStringMap final : public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DatasetDOMStringMap);
public:
    explicit DatasetDOMStringMap(Element& element)
        : m_element(element)
    {
    }

    void ref();
    void deref();

    bool isSupportedPropertyName(const String& name) const;
    Vector<String> supportedPropertyNames() const;

    String namedItem(const AtomString& name) const;
    ExceptionOr<void> setNamedItem(const String& name, const AtomString& value);
    bool deleteNamedProperty(const String& name);

    Element& element() { return m_element.get(); }
    Ref<Element> protectedElement() const;

private:
    const AtomString* item(const String& name) const;

    WeakRef<Element, WeakPtrImplWithEventTargetData> m_element;
};

} // namespace WebCore
