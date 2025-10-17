/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#include "RTCTransformEvent.h"

#if ENABLE(WEB_RTC)

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCTransformEvent);

Ref<RTCTransformEvent> RTCTransformEvent::create(const AtomString& type, Ref<RTCRtpScriptTransformer>&& transformer, IsTrusted isTrusted)
{
    return adoptRef(*new RTCTransformEvent(type, WTFMove(transformer), isTrusted));
}

RTCTransformEvent::RTCTransformEvent(const AtomString& type, Ref<RTCRtpScriptTransformer>&& transformer, IsTrusted isTrusted)
    : Event(EventInterfaceType::RTCTransformEvent, type, { }, isTrusted)
    , m_transformer(WTFMove(transformer))
{
}

RTCRtpScriptTransformer& RTCTransformEvent::transformer()
{
    return m_transformer.get();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

