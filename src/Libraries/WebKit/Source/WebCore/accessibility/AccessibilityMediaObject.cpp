/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

#if PLATFORM(IOS_FAMILY)
#include "AccessibilityMediaObject.h"

#include "HTMLMediaElement.h"
#include "HTMLNames.h"
#include "HTMLVideoElement.h"
#include "LocalizedStrings.h"


namespace WebCore {
    
using namespace HTMLNames;

AccessibilityMediaObject::AccessibilityMediaObject(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityMediaObject::~AccessibilityMediaObject() = default;

Ref<AccessibilityMediaObject> AccessibilityMediaObject::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityMediaObject(axID, renderer));
}

bool AccessibilityMediaObject::computeIsIgnored() const
{
    return isIgnoredByDefault();
}

HTMLMediaElement* AccessibilityMediaObject::mediaElement() const
{
    return dynamicDowncast<HTMLMediaElement>(node());
}

String AccessibilityMediaObject::stringValue() const
{
    if (HTMLMediaElement* element = mediaElement())
        return localizedMediaTimeDescription(element->currentTime());
    return AccessibilityRenderObject::stringValue();
}

String AccessibilityMediaObject::interactiveVideoDuration() const
{
    if (HTMLMediaElement* element = mediaElement())
        return localizedMediaTimeDescription(element->duration());
    return String();
}
    
void AccessibilityMediaObject::mediaSeek(AXSeekDirection direction)
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return;
    
    // Step 5% each time.
    const double seekStep = .05;
    double current = element->currentTime();
    double duration = element->duration();
    double timeDelta = ceil(duration * seekStep);

    double time = direction == AXSeekDirection::Forward ? std::min(current + timeDelta, duration) : std::max(current - timeDelta, 0.0);
    element->setCurrentTime(time);
}

void AccessibilityMediaObject::toggleMute()
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return;
    
    element->setMuted(!element->muted());
}

void AccessibilityMediaObject::increment()
{
    mediaSeek(AXSeekDirection::Forward);
}

void AccessibilityMediaObject::decrement()
{
    mediaSeek(AXSeekDirection::Backward);
}

bool AccessibilityMediaObject::press()
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return false;
    
    // We can safely call the internal togglePlayState method, which doesn't check restrictions,
    // because this method is only called from user interaction.
    element->togglePlayState();
    return true;
}

bool AccessibilityMediaObject::isPlaying() const
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return false;
    
    return element->isPlaying();
}

bool AccessibilityMediaObject::isMuted() const
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return false;
    
    return element->muted();
}

bool AccessibilityMediaObject::isAutoplayEnabled() const
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return false;
    
    return element->autoplay();
}

bool AccessibilityMediaObject::isPlayingInline() const
{
    HTMLMediaElement* element = mediaElement();
    if (!element)
        return false;
    
    return !element->mediaSession().requiresFullscreenForVideoPlayback();
}

void AccessibilityMediaObject::enterFullscreen() const
{
    if (RefPtr element = dynamicDowncast<HTMLVideoElement>(node()))
        element->enterFullscreen();
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
