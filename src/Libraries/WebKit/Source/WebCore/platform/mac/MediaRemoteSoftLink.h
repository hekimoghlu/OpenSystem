/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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

#include <pal/spi/mac/MediaRemoteSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(WebCore, MediaRemote)

SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteGetLocalOrigin, MROriginRef, (), ())
#define MRMediaRemoteGetLocalOrigin softLink_MediaRemote_MRMediaRemoteGetLocalOrigin
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteAddAsyncCommandHandlerBlock, void*, (MRMediaRemoteAsyncCommandHandlerBlock block), (block))
#define MRMediaRemoteAddAsyncCommandHandlerBlock softLink_MediaRemote_MRMediaRemoteAddAsyncCommandHandlerBlock
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteRemoveCommandHandlerBlock, void, (void* observer), (observer))
#define MRMediaRemoteRemoveCommandHandlerBlock softLink_MediaRemote_MRMediaRemoteRemoveCommandHandlerBlock
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetSupportedCommands, void, (CFArrayRef commands, MROriginRef origin, dispatch_queue_t replyQ, void(^completion)(MRMediaRemoteError err)), (commands, origin, replyQ, completion))
#define MRMediaRemoteSetSupportedCommands softLink_MediaRemote_MRMediaRemoteSetSupportedCommands
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingVisibility, void, (MROriginRef origin, MRNowPlayingClientVisibility visibility), (origin, visibility))
#define MRMediaRemoteSetNowPlayingVisibility softLink_MediaRemote_MRMediaRemoteSetNowPlayingVisibility
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteCommandInfoCreate, MRMediaRemoteCommandInfoRef, (CFAllocatorRef allocator), (allocator));
#define MRMediaRemoteCommandInfoCreate softLink_MediaRemote_MRMediaRemoteCommandInfoCreate
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetCommand, void, (MRMediaRemoteCommandInfoRef commandInfo, MRMediaRemoteCommand command), (commandInfo, command))
#define MRMediaRemoteCommandInfoSetCommand softLink_MediaRemote_MRMediaRemoteCommandInfoSetCommand
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetEnabled, void, (MRMediaRemoteCommandInfoRef commandInfo, Boolean enabled), (commandInfo, enabled))
#define MRMediaRemoteCommandInfoSetEnabled softLink_MediaRemote_MRMediaRemoteCommandInfoSetEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetOptions, void, (MRMediaRemoteCommandInfoRef commandInfo, CFDictionaryRef options), (commandInfo, options))
#define MRMediaRemoteCommandInfoSetOptions softLink_MediaRemote_MRMediaRemoteCommandInfoSetOptions
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetCanBeNowPlayingApplication, Boolean, (Boolean flag), (flag))
#define MRMediaRemoteSetCanBeNowPlayingApplication softLink_MediaRemote_MRMediaRemoteSetCanBeNowPlayingApplication
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingInfo, void, (CFDictionaryRef info), (info))
#define MRMediaRemoteSetNowPlayingInfo softLink_MediaRemote_MRMediaRemoteSetNowPlayingInfo
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingInfoWithMergePolicy, void, (CFDictionaryRef info, MRMediaRemoteMergePolicy mergePolicy), (info, mergePolicy))
#define MRMediaRemoteSetNowPlayingInfoWithMergePolicy softLink_MediaRemote_MRMediaRemoteSetNowPlayingInfoWithMergePolicy
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingApplicationPlaybackStateForOrigin, void, (MROriginRef origin, MRPlaybackState playbackState, dispatch_queue_t replyQ, void(^completion)(MRMediaRemoteError)), (origin, playbackState, replyQ, completion))
#define MRMediaRemoteSetNowPlayingApplicationPlaybackStateForOrigin softLink_MediaRemote_MRMediaRemoteSetNowPlayingApplicationPlaybackStateForOrigin
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteSetParentApplication, void, (MROriginRef origin, CFStringRef parentAppDisplayID), (origin, parentAppDisplayID))
#define MRMediaRemoteSetParentApplication softLink_MediaRemote_MRMediaRemoteSetParentApplication
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoTitle, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoTitle get_MediaRemote_kMRMediaRemoteNowPlayingInfoTitle()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtist, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtist get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtist()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoAlbum, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoAlbum get_MediaRemote_kMRMediaRemoteNowPlayingInfoAlbum()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkData, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtworkData get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtworkData()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkDataHeight, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtworkDataHeight get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtworkDataHeight()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkDataWidth, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtworkDataWidth get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtworkDataWidth()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkMIMEType, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtworkMIMEType get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtworkMIMEType()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoDuration, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoArtworkIdentifier get_MediaRemote_kMRMediaRemoteNowPlayingInfoArtworkIdentifier()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkIdentifier, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoDuration get_MediaRemote_kMRMediaRemoteNowPlayingInfoDuration()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoElapsedTime, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoElapsedTime get_MediaRemote_kMRMediaRemoteNowPlayingInfoElapsedTime()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoPlaybackRate, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoPlaybackRate get_MediaRemote_kMRMediaRemoteNowPlayingInfoPlaybackRate()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteOptionPlaybackPosition, CFStringRef);
#define kMRMediaRemoteOptionPlaybackPosition get_MediaRemote_kMRMediaRemoteOptionPlaybackPosition()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoUniqueIdentifier, CFStringRef);
#define kMRMediaRemoteNowPlayingInfoUniqueIdentifier get_MediaRemote_kMRMediaRemoteNowPlayingInfoUniqueIdentifier()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteOptionSkipInterval, CFStringRef);
#define kMRMediaRemoteOptionSkipInterval get_MediaRemote_kMRMediaRemoteOptionSkipInterval()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, MediaRemote, kMRMediaRemoteCommandInfoPreferredIntervalsKey, CFStringRef);
#define kMRMediaRemoteCommandInfoPreferredIntervalsKey get_MediaRemote_kMRMediaRemoteOptionSkipInterval()

#if PLATFORM(IOS_FAMILY)
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, MediaRemote, MRMediaRemoteCopyPickableRoutes, CFArrayRef, (), ())
#define MRMediaRemoteCopyPickableRoutes softLink_MediaRemote_MRMediaRemoteCopyPickableRoutes
#endif

#if USE(NOW_PLAYING_ACTIVITY_SUPPRESSION)
SOFT_LINK_CLASS_FOR_HEADER(WebCore, MRUIControllerProvider);
#endif
