/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#include "MediaQueryListEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaQueryListEvent);

MediaQueryListEvent::MediaQueryListEvent(const AtomString& type, const String& media, bool matches)
    : Event(EventInterfaceType::MediaQueryListEvent, type, CanBubble::No, IsCancelable::No)
    , m_media(media)
    , m_matches(matches)
{
}

MediaQueryListEvent::MediaQueryListEvent(const AtomString& type, const Init& init, IsTrusted isTrusted)
    : Event(EventInterfaceType::MediaQueryListEvent, type, init, isTrusted)
    , m_media(init.media)
    , m_matches(init.matches)
{
}

Ref<MediaQueryListEvent> MediaQueryListEvent::create(const AtomString& type, const String& media, bool matches)
{
    return adoptRef(*new MediaQueryListEvent(type, media, matches));
}

Ref<MediaQueryListEvent> MediaQueryListEvent::create(const AtomString& type, const Init& init, IsTrusted isTrusted)
{
    return adoptRef(*new MediaQueryListEvent(type, init, isTrusted));
}

} // namespace WebCore
