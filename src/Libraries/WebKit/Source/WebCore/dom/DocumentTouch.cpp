/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#include "DocumentTouch.h"

#if ENABLE(TOUCH_EVENTS)

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "Touch.h"
#include "TouchList.h"
#include "WindowProxy.h"

namespace WebCore {

Ref<Touch> DocumentTouch::createTouch(Document& document, RefPtr<WindowProxy>&& window, EventTarget* target, int identifier, int pageX, int pageY, int screenX, int screenY, int radiusX, int radiusY, float rotationAngle, float force)
{
    RefPtr<LocalFrame> frame;
    if (window)
        frame = dynamicDowncast<LocalFrame>(window->frame());
    if (!frame)
        frame = document.frame();

    // FIXME: It's not clear from the documentation at
    // http://developer.apple.com/library/safari/#documentation/UserExperience/Reference/DocumentAdditionsReference/DocumentAdditions/DocumentAdditions.html
    // when this method should throw and nor is it by inspection of iOS behavior. It would be nice to verify any cases where it throws under iOS
    // and implement them here. See https://bugs.webkit.org/show_bug.cgi?id=47819
    return Touch::create(frame.get(), target, identifier, screenX, screenY, pageX, pageY, radiusX, radiusY, rotationAngle, force);
}

Ref<TouchList> DocumentTouch::createTouchList(Document&, FixedVector<std::reference_wrapper<Touch>>&& touches)
{
    return TouchList::create(WTFMove(touches));
}

}

#endif
