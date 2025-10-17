/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "PictureInPictureWindow.h"

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "Event.h"
#include "EventNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PictureInPictureWindow);

Ref<PictureInPictureWindow> PictureInPictureWindow::create(Document& document)
{
    auto window = adoptRef(*new PictureInPictureWindow(document));
    window->suspendIfNeeded();
    return window;
}

PictureInPictureWindow::PictureInPictureWindow(Document& document)
    : ActiveDOMObject(document)
{
}

PictureInPictureWindow::~PictureInPictureWindow() = default;

void PictureInPictureWindow::setSize(const IntSize& size)
{
    if (width() == size.width() && height() == size.height())
        return;
    
    m_size = size;
    queueTaskToDispatchEvent(*this, TaskSource::MediaElement, Event::create(eventNames().resizeEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void PictureInPictureWindow::close()
{
    m_size = { 0, 0 };
}

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
