/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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

#if ENABLE(IOS_TOUCH_EVENTS)
#include <WebKitAdditions/DocumentTouchIOS.h>
#elif ENABLE(TOUCH_EVENTS)

#include <functional>
#include <wtf/FixedVector.h>
#include <wtf/Forward.h>

namespace WebCore {

class Document;
class EventTarget;
class Touch;
class TouchList;
class WindowProxy;

class DocumentTouch {
public:
    static Ref<Touch> createTouch(Document&, RefPtr<WindowProxy>&&, EventTarget*, int identifier, int pageX, int pageY, int screenX, int screenY, int radiusX, int radiusY, float rotationAngle, float force);
    static Ref<TouchList> createTouchList(Document&, FixedVector<std::reference_wrapper<Touch>>&&);
};

}

#endif
