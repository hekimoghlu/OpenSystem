/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

#include <unknwn.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

    class AccessibilityObject;
 
    class AccessibilityObjectWrapper : public IUnknown {
    public:
        // IUnknown
        virtual HRESULT STDMETHODCALLTYPE QueryInterface(_In_ REFIID riid, _COM_Outptr_ void** ppvObject) = 0;        
        virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
        virtual ULONG STDMETHODCALLTYPE Release(void) = 0;

        virtual void detach() = 0;
        bool attached() const { return m_object; }
        AccessibilityObject* accessibilityObject() const { return m_object; }

        WEBCORE_EXPORT void accessibilityAttributeValue(const AtomString&, VARIANT*);

    protected:
        AccessibilityObjectWrapper(AccessibilityObject* obj) : m_object(obj) { }
        AccessibilityObjectWrapper() : m_object(nullptr) { }

        AccessibilityObject* m_object;
    };

} // namespace WebCore
