/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include "PictureInPictureEvent.h"

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "PictureInPictureWindow.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PictureInPictureEvent);

Ref<PictureInPictureEvent> PictureInPictureEvent::create(const AtomString& type, Init&& init, IsTrusted isTrusted)
{
    return adoptRef(*new PictureInPictureEvent(type, WTFMove(init), isTrusted));
}

PictureInPictureEvent::PictureInPictureEvent(const AtomString& type, Init&& init, IsTrusted isTrusted)
    : Event(EventInterfaceType::PictureInPictureEvent, type, init, isTrusted)
    , m_pictureInPictureWindow(init.pictureInPictureWindow.releaseNonNull())
{
}

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
