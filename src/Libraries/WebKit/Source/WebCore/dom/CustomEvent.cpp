/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#include "CustomEvent.h"

#include <JavaScriptCore/JSCInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CustomEvent);

inline CustomEvent::CustomEvent(IsTrusted isTrusted)
    : Event(EventInterfaceType::CustomEvent, isTrusted)
{
}

inline CustomEvent::CustomEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::CustomEvent, type, initializer, isTrusted)
    , m_detail(initializer.detail)
{
}

CustomEvent::~CustomEvent() = default;

Ref<CustomEvent> CustomEvent::create(IsTrusted isTrusted)
{
    return adoptRef(*new CustomEvent(isTrusted));
}

Ref<CustomEvent> CustomEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new CustomEvent(type, initializer, isTrusted));
}

void CustomEvent::initCustomEvent(const AtomString& type, bool canBubble, bool cancelable, JSC::JSValue detail)
{
    if (isBeingDispatched())
        return;

    initEvent(type, canBubble, cancelable);

    // FIXME: This code is wrong: we should emit a write-barrier. Otherwise, GC can collect it.
    // https://bugs.webkit.org/show_bug.cgi?id=236353
    m_detail.setWeakly(detail);
    m_cachedDetail.clear();
}

} // namespace WebCore
