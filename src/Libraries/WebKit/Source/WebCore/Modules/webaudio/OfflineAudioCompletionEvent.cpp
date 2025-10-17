/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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

#if ENABLE(WEB_AUDIO)

#include "OfflineAudioCompletionEvent.h"

#include "AudioBuffer.h"
#include "EventNames.h"
#include "OfflineAudioCompletionEventInit.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/TypedArrayAdaptors.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OfflineAudioCompletionEvent);

Ref<OfflineAudioCompletionEvent> OfflineAudioCompletionEvent::create(Ref<AudioBuffer>&& renderedBuffer)
{
    return adoptRef(*new OfflineAudioCompletionEvent(WTFMove(renderedBuffer)));
}

Ref<OfflineAudioCompletionEvent> OfflineAudioCompletionEvent::create(const AtomString& eventType, OfflineAudioCompletionEventInit&& init)
{
    RELEASE_ASSERT(init.renderedBuffer);
    return adoptRef(*new OfflineAudioCompletionEvent(eventType, WTFMove(init)));
}

OfflineAudioCompletionEvent::OfflineAudioCompletionEvent(Ref<AudioBuffer>&& renderedBuffer)
    : Event(EventInterfaceType::OfflineAudioCompletionEvent, eventNames().completeEvent, CanBubble::Yes, IsCancelable::No)
    , m_renderedBuffer(WTFMove(renderedBuffer))
{
}

OfflineAudioCompletionEvent::OfflineAudioCompletionEvent(const AtomString& eventType, OfflineAudioCompletionEventInit&& init)
    : Event(EventInterfaceType::OfflineAudioCompletionEvent, eventType, init, IsTrusted::No)
    , m_renderedBuffer(init.renderedBuffer.releaseNonNull())
{
}

OfflineAudioCompletionEvent::~OfflineAudioCompletionEvent() = default;

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
