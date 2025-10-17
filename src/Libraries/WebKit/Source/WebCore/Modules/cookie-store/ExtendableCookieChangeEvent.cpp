/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#include "ExtendableCookieChangeEvent.h"

#include "CookieListItem.h"
#include "ExtendableCookieChangeEventInit.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ExtendableCookieChangeEvent);

Ref<ExtendableCookieChangeEvent> ExtendableCookieChangeEvent::create(const AtomString& type, ExtendableCookieChangeEventInit&& eventInitDict, IsTrusted isTrusted)
{
    return adoptRef(*new ExtendableCookieChangeEvent(type, WTFMove(eventInitDict), isTrusted));
}

ExtendableCookieChangeEvent::ExtendableCookieChangeEvent(const AtomString& type, ExtendableCookieChangeEventInit&& eventInitDict, IsTrusted isTrusted)
    : ExtendableEvent(EventInterfaceType::ExtendableCookieChangeEvent, type, eventInitDict, isTrusted)
    , m_changed(WTFMove(eventInitDict.changed))
    , m_deleted(WTFMove(eventInitDict.deleted))
{ }

ExtendableCookieChangeEvent::~ExtendableCookieChangeEvent() = default;

} // namespace WebCore
