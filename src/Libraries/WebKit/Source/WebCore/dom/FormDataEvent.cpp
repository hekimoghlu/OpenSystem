/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#include "FormDataEvent.h"

#include "DOMFormData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FormDataEvent);

Ref<FormDataEvent> FormDataEvent::create(const AtomString& eventType, Init&& init)
{
    return adoptRef(*new FormDataEvent(eventType, WTFMove(init)));
}

Ref<FormDataEvent> FormDataEvent::create(const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, Ref<DOMFormData>&& formData)
{
    return adoptRef(*new FormDataEvent(eventType, canBubble, isCancelable, isComposed, WTFMove(formData)));
}

FormDataEvent::FormDataEvent(const AtomString& eventType, Init&& init)
    : Event(EventInterfaceType::FormDataEvent, eventType, init, IsTrusted::No)
    , m_formData(init.formData.releaseNonNull())
{
}

FormDataEvent::FormDataEvent(const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, Ref<DOMFormData>&& formData)
    : Event(EventInterfaceType::FormDataEvent, eventType, canBubble, isCancelable, isComposed)
    , m_formData(WTFMove(formData))
{
}

} // namespace WebCore
