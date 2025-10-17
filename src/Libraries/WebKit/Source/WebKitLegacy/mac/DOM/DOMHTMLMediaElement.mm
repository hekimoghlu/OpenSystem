/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#if ENABLE(VIDEO)

#import "DOMHTMLMediaElement.h"

#import "DOMInternal.h"
#import "DOMMediaErrorInternal.h"
#import "DOMNodeInternal.h"
#import "DOMTimeRangesInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLMediaElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/MediaError.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/TimeRanges.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLMediaElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLMediaElement

- (DOMMediaError *)error
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->error()));
}

- (NSString *)src
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::srcAttr).string();
}

- (void)setSrc:(NSString *)newSrc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::srcAttr, newSrc);
}

- (NSString *)currentSrc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->currentSrc().string();
}

- (NSString *)crossOrigin
{
    WebCore::JSMainThreadNullState state;
    return IMPL->crossOrigin();
}

- (void)setCrossOrigin:(NSString *)newCrossOrigin
{
    WebCore::JSMainThreadNullState state;
    IMPL->setCrossOrigin(newCrossOrigin);
}

- (unsigned short)networkState
{
    WebCore::JSMainThreadNullState state;
    return IMPL->networkState();
}

- (NSString *)preload
{
    WebCore::JSMainThreadNullState state;
    return IMPL->preload();
}

- (void)setPreload:(NSString *)newPreload
{
    WebCore::JSMainThreadNullState state;
    IMPL->setPreload(newPreload);
}

- (DOMTimeRanges *)buffered
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->buffered()));
}

- (unsigned short)readyState
{
    WebCore::JSMainThreadNullState state;
    return IMPL->readyState();
}

- (BOOL)seeking
{
    WebCore::JSMainThreadNullState state;
    return IMPL->seeking();
}

- (double)currentTime
{
    WebCore::JSMainThreadNullState state;
    return IMPL->currentTime();
}

- (void)setCurrentTime:(double)newCurrentTime
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setCurrentTimeForBindings(newCurrentTime));
}

- (double)duration
{
    WebCore::JSMainThreadNullState state;
    return IMPL->duration();
}

- (BOOL)paused
{
    WebCore::JSMainThreadNullState state;
    return IMPL->paused();
}

- (double)defaultPlaybackRate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->defaultPlaybackRate();
}

- (void)setDefaultPlaybackRate:(double)newDefaultPlaybackRate
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDefaultPlaybackRate(newDefaultPlaybackRate);
}

- (double)playbackRate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->playbackRate();
}

- (void)setPlaybackRate:(double)newPlaybackRate
{
    WebCore::JSMainThreadNullState state;
    IMPL->setPlaybackRate(newPlaybackRate);
}

- (DOMTimeRanges *)played
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->played()));
}

- (DOMTimeRanges *)seekable
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->seekable()));
}

- (BOOL)ended
{
    WebCore::JSMainThreadNullState state;
    return IMPL->ended();
}

- (BOOL)autoplay
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::autoplayAttr);
}

- (void)setAutoplay:(BOOL)newAutoplay
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::autoplayAttr, newAutoplay);
}

- (BOOL)loop
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::loopAttr);
}

- (void)setLoop:(BOOL)newLoop
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::loopAttr, newLoop);
}

- (BOOL)controls
{
    WebCore::JSMainThreadNullState state;
    return IMPL->controls();
}

- (void)setControls:(BOOL)newControls
{
    WebCore::JSMainThreadNullState state;
    IMPL->setControls(newControls);
}

- (double)volume
{
    WebCore::JSMainThreadNullState state;
    return IMPL->volume();
}

- (void)setVolume:(double)newVolume
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setVolume(newVolume));
}

- (BOOL)muted
{
    WebCore::JSMainThreadNullState state;
    return IMPL->muted();
}

- (void)setMuted:(BOOL)newMuted
{
    WebCore::JSMainThreadNullState state;
    IMPL->setMuted(newMuted);
}

- (BOOL)defaultMuted
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::mutedAttr);
}

- (void)setDefaultMuted:(BOOL)newDefaultMuted
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::mutedAttr, newDefaultMuted);
}

- (BOOL)webkitPreservesPitch
{
    WebCore::JSMainThreadNullState state;
    return IMPL->preservesPitch();
}

- (void)setWebkitPreservesPitch:(BOOL)newWebkitPreservesPitch
{
    WebCore::JSMainThreadNullState state;
    IMPL->setPreservesPitch(newWebkitPreservesPitch);
}

- (BOOL)webkitHasClosedCaptions
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasClosedCaptions();
}

- (BOOL)webkitClosedCaptionsVisible
{
    WebCore::JSMainThreadNullState state;
    return IMPL->closedCaptionsVisible();
}

- (void)setWebkitClosedCaptionsVisible:(BOOL)newWebkitClosedCaptionsVisible
{
    WebCore::JSMainThreadNullState state;
    IMPL->setClosedCaptionsVisible(newWebkitClosedCaptionsVisible);
}

- (NSString *)mediaGroup
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::mediagroupAttr);
}

- (void)setMediaGroup:(NSString *)newMediaGroup
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::mediagroupAttr, newMediaGroup);
}

- (void)load
{
    WebCore::JSMainThreadNullState state;
    IMPL->load();
}

- (NSString *)canPlayType:(NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->canPlayType(type);
}

- (NSTimeInterval)getStartDate
{
    WebCore::JSMainThreadNullState state;
    return kit(IMPL->getStartDate());
}

- (void)play
{
    WebCore::JSMainThreadNullState state;
    IMPL->play();
}

- (void)pause
{
    WebCore::JSMainThreadNullState state;
    IMPL->pause();
}

- (void)fastSeek:(double)time
{
    WebCore::JSMainThreadNullState state;
    IMPL->fastSeek(time);
}

@end

#endif // ENABLE(VIDEO)

#undef IMPL
