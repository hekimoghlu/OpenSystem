/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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

#if ENABLE(VIDEO) && USE(AVFOUNDATION)

#import "InbandTextTrackPrivateAVFObjC.h"

#import "FloatConversion.h"
#import "InbandTextTrackPrivate.h"
#import "InbandTextTrackPrivateAVF.h"
#import "Logging.h"
#import <AVFoundation/AVMediaSelectionGroup.h>
#import <AVFoundation/AVMetadataItem.h>
#import <AVFoundation/AVPlayer.h>
#import <AVFoundation/AVPlayerItem.h>
#import <AVFoundation/AVPlayerItemOutput.h>
#import <objc/runtime.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

@class AVMediaSelectionOption;
@interface AVMediaSelectionOption (WebKitInternal)
- (id)optionID;
@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InbandTextTrackPrivateAVFObjC);

InbandTextTrackPrivateAVFObjC::InbandTextTrackPrivateAVFObjC(AVFInbandTrackParent* player, AVMediaSelectionGroup *group, AVMediaSelectionOption *selection, TrackID trackID, InbandTextTrackPrivate::CueFormat format)
    : InbandTextTrackPrivateAVF(player, trackID, format)
    , m_mediaSelectionGroup(group)
    , m_mediaSelectionOption(selection)
{
}

void InbandTextTrackPrivateAVFObjC::disconnect()
{
    m_mediaSelectionGroup = 0;
    m_mediaSelectionOption = 0;
    InbandTextTrackPrivateAVF::disconnect();
}

InbandTextTrackPrivate::Kind InbandTextTrackPrivateAVFObjC::kind() const
{
    if (!m_mediaSelectionOption)
        return Kind::None;

    NSString *mediaType = [m_mediaSelectionOption mediaType];
    
    if ([mediaType isEqualToString:AVMediaTypeClosedCaption])
        return Kind::Captions;
    if ([mediaType isEqualToString:AVMediaTypeSubtitle]) {

        if ([m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicContainsOnlyForcedSubtitles])
            return Kind::Forced;

        // An "SDH" track is a subtitle track created for the deaf or hard-of-hearing. "captions" in WebVTT are
        // "labeled as appropriate for the hard-of-hearing", so tag SDH sutitles as "captions".
        if ([m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicTranscribesSpokenDialogForAccessibility])
            return Kind::Captions;
        if ([m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicDescribesMusicAndSoundForAccessibility])
            return Kind::Captions;
        
        return Kind::Subtitles;
    }

    return Kind::Captions;
}

bool InbandTextTrackPrivateAVFObjC::isClosedCaptions() const
{
    if (!m_mediaSelectionOption)
        return false;
    
    return [[m_mediaSelectionOption mediaType] isEqualToString:AVMediaTypeClosedCaption];
}

bool InbandTextTrackPrivateAVFObjC::isSDH() const
{
    if (!m_mediaSelectionOption)
        return false;
    
    if (![[m_mediaSelectionOption mediaType] isEqualToString:AVMediaTypeSubtitle])
        return false;

    if ([m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicTranscribesSpokenDialogForAccessibility] && [m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicDescribesMusicAndSoundForAccessibility])
        return true;

    return false;
}
    
bool InbandTextTrackPrivateAVFObjC::containsOnlyForcedSubtitles() const
{
    if (!m_mediaSelectionOption)
        return false;
    
    return [m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicContainsOnlyForcedSubtitles];
}

bool InbandTextTrackPrivateAVFObjC::isMainProgramContent() const
{
    if (!m_mediaSelectionOption)
        return false;
    
    return [m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicIsMainProgramContent];
}

bool InbandTextTrackPrivateAVFObjC::isEasyToRead() const
{
    if (!m_mediaSelectionOption)
        return false;

    return [m_mediaSelectionOption hasMediaCharacteristic:AVMediaCharacteristicEasyToRead];
}

AtomString InbandTextTrackPrivateAVFObjC::label() const
{
    if (!m_mediaSelectionOption)
        return emptyAtom();

    NSString *title = 0;

    NSArray *titles = [PAL::getAVMetadataItemClass() metadataItemsFromArray:[m_mediaSelectionOption commonMetadata] withKey:AVMetadataCommonKeyTitle keySpace:AVMetadataKeySpaceCommon];
    if ([titles count]) {
        // If possible, return a title in one of the user's preferred languages.
        NSArray *titlesForPreferredLanguages = [PAL::getAVMetadataItemClass() metadataItemsFromArray:titles filteredAndSortedAccordingToPreferredLanguages:[NSLocale preferredLanguages]];
        if ([titlesForPreferredLanguages count])
            title = [[titlesForPreferredLanguages objectAtIndex:0] stringValue];

        if (!title)
            title = [[titles objectAtIndex:0] stringValue];
    }

    return title ? AtomString(title) : emptyAtom();
}

AtomString InbandTextTrackPrivateAVFObjC::language() const
{
    if (!m_mediaSelectionOption)
        return emptyAtom();

    return [[m_mediaSelectionOption locale] localeIdentifier];
}

bool InbandTextTrackPrivateAVFObjC::isDefault() const
{
    return [m_mediaSelectionGroup defaultOption] == m_mediaSelectionOption.get();
}

} // namespace WebCore

#endif
