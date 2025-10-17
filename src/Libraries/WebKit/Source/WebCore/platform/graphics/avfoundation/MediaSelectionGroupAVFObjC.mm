/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#import "config.h"
#import "MediaSelectionGroupAVFObjC.h"

#if ENABLE(VIDEO)

#import <AVFoundation/AVAsset.h>
#import <AVFoundation/AVPlayerItem.h>
#import <objc/runtime.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <wtf/Language.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/WTFString.h>

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
#import <MediaAccessibility/MediaAccessibility.h>
#import "MediaAccessibilitySoftLink.h"
#endif

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

Ref<MediaSelectionOptionAVFObjC> MediaSelectionOptionAVFObjC::create(MediaSelectionGroupAVFObjC& group, AVMediaSelectionOption *option)
{
    return adoptRef(*new MediaSelectionOptionAVFObjC(group, option));
}

MediaSelectionOptionAVFObjC::MediaSelectionOptionAVFObjC(MediaSelectionGroupAVFObjC& group, AVMediaSelectionOption *option)
    : m_group(&group)
    , m_mediaSelectionOption(option)
{
}

void MediaSelectionOptionAVFObjC::setSelected(bool selected)
{
    if (!m_group)
        return;

    if (selected == this->selected())
        return;

    m_group->setSelectedOption(selected ? this : nullptr);
}

bool MediaSelectionOptionAVFObjC::selected() const
{
    if (!m_group)
        return false;
    return this == m_group->selectedOption();
}

int MediaSelectionOptionAVFObjC::index() const
{
    if (!m_group)
        return 0;

    return [[m_group->avMediaSelectionGroup() options] indexOfObject:m_mediaSelectionOption.get()];
}

AVAssetTrack* MediaSelectionOptionAVFObjC::assetTrack() const
{
    if ([m_mediaSelectionOption respondsToSelector:@selector(track)])
        return [m_mediaSelectionOption track];
    return nil;
}

Ref<MediaSelectionGroupAVFObjC> MediaSelectionGroupAVFObjC::create(AVPlayerItem *item, AVMediaSelectionGroup *group, const Vector<String>& characteristics)
{
    return adoptRef(*new MediaSelectionGroupAVFObjC(item, group, characteristics));
}

MediaSelectionGroupAVFObjC::MediaSelectionGroupAVFObjC(AVPlayerItem *item, AVMediaSelectionGroup *group, const Vector<String>& characteristics)
    : m_playerItem(item)
    , m_mediaSelectionGroup(group)
    , m_selectionTimer(*this, &MediaSelectionGroupAVFObjC::selectionTimerFired)
{
    updateOptions(characteristics);
}

MediaSelectionGroupAVFObjC::~MediaSelectionGroupAVFObjC()
{
    for (auto& option : m_options.values())
        option->clearGroup();
}

void MediaSelectionGroupAVFObjC::updateOptions(const Vector<String>& characteristics)
{
    RetainPtr<NSSet> newAVOptions = adoptNS([[NSSet alloc] initWithArray:[PAL::getAVMediaSelectionGroupClass() playableMediaSelectionOptionsFromArray:[m_mediaSelectionGroup options]]]);
    RetainPtr<NSMutableSet> oldAVOptions = adoptNS([[NSMutableSet alloc] initWithCapacity:m_options.size()]);
    for (auto& avOption : m_options.keys())
        [oldAVOptions addObject:(__bridge AVMediaSelectionOption *)avOption];

    RetainPtr<NSMutableSet> addedAVOptions = adoptNS([newAVOptions mutableCopy]);
    [addedAVOptions minusSet:oldAVOptions.get()];

    RetainPtr<NSMutableSet> removedAVOptions = adoptNS([oldAVOptions mutableCopy]);
    [removedAVOptions minusSet:newAVOptions.get()];

    for (AVMediaSelectionOption* removedAVOption in removedAVOptions.get()) {
        if (m_selectedOption && removedAVOption == m_selectedOption->avMediaSelectionOption())
            m_selectedOption = nullptr;

        m_options.remove((__bridge CFTypeRef)removedAVOption);
    }
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    AVMediaSelectionOption* selectedOption = [m_playerItem selectedMediaOptionInMediaSelectionGroup:m_mediaSelectionGroup.get()];
ALLOW_DEPRECATED_DECLARATIONS_END
    for (AVMediaSelectionOption* addedAVOption in addedAVOptions.get()) {
        auto addedOption = MediaSelectionOptionAVFObjC::create(*this, addedAVOption);
        if (addedAVOption == selectedOption)
            m_selectedOption = addedOption.ptr();
        m_options.set((__bridge CFTypeRef)addedAVOption, WTFMove(addedOption));
    }

    if (!m_shouldSelectOptionAutomatically)
        return;

    NSArray* filteredOptions = [PAL::getAVMediaSelectionGroupClass() mediaSelectionOptionsFromArray:[m_mediaSelectionGroup options]
        filteredAndSortedAccordingToPreferredLanguages:createNSArray(userPreferredLanguages(ShouldMinimizeLanguages::No)).get()];

    if (![filteredOptions count] && characteristics.isEmpty())
        return;

    // If no options match our language selection, search for matching characteristics across all the group's options
    if (![filteredOptions count])
        filteredOptions = [m_mediaSelectionGroup options];

    NSArray* optionsWithCharacteristics = [PAL::getAVMediaSelectionGroupClass() mediaSelectionOptionsFromArray:filteredOptions withMediaCharacteristics:createNSArray(characteristics).get()];
    if (optionsWithCharacteristics && [optionsWithCharacteristics count])
        filteredOptions = optionsWithCharacteristics;

    if (![filteredOptions count])
        return;

    AVMediaSelectionOption* preferredOption = [filteredOptions objectAtIndex:0];
    if (m_selectedOption && m_selectedOption->avMediaSelectionOption() == preferredOption)
        return;

    ASSERT(m_options.contains((__bridge CFTypeRef)preferredOption));
    m_selectedOption = m_options.get((__bridge CFTypeRef)preferredOption);
    m_selectionTimer.startOneShot(0_s);
}

void MediaSelectionGroupAVFObjC::setSelectedOption(MediaSelectionOptionAVFObjC* option)
{
    if (m_selectedOption == option)
        return;

    m_shouldSelectOptionAutomatically = false;
    m_selectedOption = option;
    if (m_selectionTimer.isActive())
        m_selectionTimer.stop();
    m_selectionTimer.startOneShot(0_s);
}

void MediaSelectionGroupAVFObjC::selectionTimerFired()
{
    @try {
        [m_playerItem selectMediaOption:(m_selectedOption ? m_selectedOption->avMediaSelectionOption() : nil) inMediaSelectionGroup:m_mediaSelectionGroup.get()];
    } @catch(NSException *exception) {
        WTFReportError(__FILE__, __LINE__, WTF_PRETTY_FUNCTION, "exception thrown from -selectMediaOption:inMediaSelectionGroup: %s", exception.name.UTF8String);
    }
}

}

#endif
