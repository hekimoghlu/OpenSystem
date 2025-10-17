/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#import "InbandChapterTrackPrivateAVFObjC.h"

#if ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#import "ISOVTTCue.h"
#import "InbandTextTrackPrivateClient.h"
#import <AVFoundation/AVMetadataItem.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/StringBuilder.h>
#import <wtf/text/WTFString.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InbandChapterTrackPrivateAVFObjC);

InbandChapterTrackPrivateAVFObjC::InbandChapterTrackPrivateAVFObjC(RetainPtr<NSLocale> locale, TrackID trackID)
    : InbandTextTrackPrivate(CueFormat::WebVTT)
    , m_locale(WTFMove(locale))
    , m_id(trackID)
{
    setMode(Mode::Hidden);
}

void InbandChapterTrackPrivateAVFObjC::processChapters(RetainPtr<NSArray<AVTimedMetadataGroup *>> chapters)
{
    if (!hasClients())
        return;

    auto identifier = LOGIDENTIFIER;
    auto createChapterCue = ([this, identifier] (AVMetadataItem *item, int chapterNumber) mutable {
        if (!hasClients())
            return;
        ASSERT(hasOneClient());
        ChapterData chapterData = { PAL::toMediaTime([item time]), PAL::toMediaTime([item duration]), [item stringValue] };
        if (m_processedChapters.contains(chapterData))
            return;
        m_processedChapters.append(chapterData);

        ISOWebVTTCue cueData = ISOWebVTTCue(PAL::toMediaTime([item time]), PAL::toMediaTime([item duration]), AtomString::number(chapterNumber), [item stringValue]);
        INFO_LOG(identifier, "created cue ", cueData);
        notifyMainThreadClient([cueData = WTFMove(cueData)](TrackPrivateBaseClient& client) mutable {
            downcast<InbandTextTrackPrivateClient>(client).parseWebVTTCueData(WTFMove(cueData));
        });
    });

    int chapterNumber = 0;
    for (AVTimedMetadataGroup *chapter in chapters.get()) {
        for (AVMetadataItem *item in [chapter items]) {
            ++chapterNumber;
            if ([item statusOfValueForKey:@"value" error:nil] == AVKeyValueStatusLoaded)
                createChapterCue(item, chapterNumber);
            else {
                [item loadValuesAsynchronouslyForKeys:@[@"value"] completionHandler:[this, protectedThis = Ref { *this }, item = retainPtr(item), createChapterCue, chapterNumber, identifier] () mutable {

                    NSError *error = nil;
                    auto keyStatus = [item statusOfValueForKey:@"value" error:&error];
                    if (error)
                        ERROR_LOG(identifier, "@\"value\" failed failed to load, status is ", (int)keyStatus, ", error = ", error);

                    if (keyStatus == AVKeyValueStatusLoaded && !error) {
                        callOnMainThread([item = WTFMove(item), protectedThis = WTFMove(protectedThis), createChapterCue = WTFMove(createChapterCue), chapterNumber] () mutable {
                            createChapterCue(item.get(), chapterNumber);
                        });
                    }
                }];
            }
        }
    }
}

AtomString InbandChapterTrackPrivateAVFObjC::language() const
{
    if (!m_language.isEmpty())
        return m_language;

    m_language = [m_locale localeIdentifier];
    return m_language;
}

} // namespace WebCore

#endif // ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))
