/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include "XRInputSourcesChangeEvent.h"

#if ENABLE(WEBXR)

#include "WebXRInputSource.h"
#include "WebXRSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XRInputSourcesChangeEvent);

Ref<XRInputSourcesChangeEvent> XRInputSourcesChangeEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new XRInputSourcesChangeEvent(type, initializer, isTrusted));
}

XRInputSourcesChangeEvent::XRInputSourcesChangeEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::XRInputSourcesChangeEvent, type, initializer, isTrusted)
    , m_session(*initializer.session)
    , m_added(initializer.added)
    , m_removed(initializer.removed)
{
}

XRInputSourcesChangeEvent::~XRInputSourcesChangeEvent() = default;

const WebXRSession& XRInputSourcesChangeEvent::session() const
{
    return m_session;
}

const Vector<Ref<WebXRInputSource>>& XRInputSourcesChangeEvent::added() const
{
    return m_added;
}

const Vector<Ref<WebXRInputSource>>& XRInputSourcesChangeEvent::removed() const
{
    return m_removed;
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
