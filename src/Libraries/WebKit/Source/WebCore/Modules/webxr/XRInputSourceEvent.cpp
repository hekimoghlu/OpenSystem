/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "XRInputSourceEvent.h"

#if ENABLE(WEBXR)

#include "WebXRFrame.h"
#include "WebXRInputSource.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XRInputSourceEvent);

Ref<XRInputSourceEvent> XRInputSourceEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new XRInputSourceEvent(type, initializer, isTrusted));
}

XRInputSourceEvent::XRInputSourceEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::XRInputSourceEvent, type, initializer, isTrusted)
    , m_frame(initializer.frame)
    , m_inputSource(initializer.inputSource)
{
    ASSERT(m_frame);
    ASSERT(m_inputSource);
}

XRInputSourceEvent::~XRInputSourceEvent() = default;

const WebXRFrame& XRInputSourceEvent::frame() const
{
    return *m_frame;
}

const WebXRInputSource& XRInputSourceEvent::inputSource() const
{
    return *m_inputSource;
}

void XRInputSourceEvent::setFrameActive(bool active)
{
    m_frame->setActive(active);
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
