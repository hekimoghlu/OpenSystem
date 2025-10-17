/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

#include "Event.h"
#include "EventInit.h"

namespace WebCore {

class DOMFormData;

class FormDataEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FormDataEvent);
public:
    struct Init : EventInit {
        RefPtr<DOMFormData> formData;
    };
        
    static Ref<FormDataEvent> create(const AtomString&, Init&&);
    static Ref<FormDataEvent> create(const AtomString&, CanBubble, IsCancelable, IsComposed, Ref<DOMFormData>&&);

    const DOMFormData& formData() const { return m_formData.get(); }
    
private:
    FormDataEvent(const AtomString&, Init&&);
    FormDataEvent(const AtomString&, CanBubble, IsCancelable, IsComposed, Ref<DOMFormData>&&);

    Ref<DOMFormData> m_formData;
};

} // namespace WebCore
