/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include "AccessibilityObjectWrapperWin.h"

#include "AXObjectCache.h"
#include "AccessibilityObject.h"
#include "BString.h"
#include "HTMLNames.h"
#include "QualifiedName.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

void AccessibilityObjectWrapper::accessibilityAttributeValue(const AtomString& attributeName, VARIANT* result)
{
    // FIXME: This should be fleshed out to match the Mac version

    m_object->updateBackingStore();

    // Not a real concept on Windows, but used heavily in WebKit accessibility testing.
    if (attributeName == "AXTitleUIElementAttribute"_s) {
        if (auto* object = m_object->titleUIElement()) {
            ASSERT(V_VT(result) == VT_EMPTY);
            V_VT(result) = VT_UNKNOWN;
            AccessibilityObjectWrapper* wrapper = object->wrapper();
            V_UNKNOWN(result) = wrapper;
            if (wrapper)
                wrapper->AddRef();
        }
        return;
    }

    // Used to find an accessible node by its element id.
    if (attributeName == "AXDOMIdentifier"_s) {
        ASSERT(V_VT(result) == VT_EMPTY);

        V_VT(result) = VT_BSTR;
        V_BSTR(result) = WebCore::BString(m_object->getAttribute(WebCore::HTMLNames::idAttr)).release();
        return;
    }

    if (attributeName == "AXSelectedTextRangeAttribute"_s) {
        ASSERT(V_VT(result) == VT_EMPTY);
        V_VT(result) = VT_BSTR;
        CharacterRange textRange = m_object->selectedTextRange();
        String range = makeString('{', textRange.location, ", "_s, textRange.length, '}');
        V_BSTR(result) = WebCore::BString(range).release();
        return;
    }
}


} // namespace WebCore
