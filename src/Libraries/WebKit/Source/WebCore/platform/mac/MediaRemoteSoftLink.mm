/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

#include <pal/spi/mac/MediaRemoteSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE(WebCore, MediaRemote)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteGetLocalOrigin, MROriginRef, (), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteAddAsyncCommandHandlerBlock, void*, (MRMediaRemoteAsyncCommandHandlerBlock block), (block))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteRemoveCommandHandlerBlock, void, (void* observer), (observer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetSupportedCommands, void, (CFArrayRef commands, MROriginRef origin, dispatch_queue_t replyQ, void(^completion)(MRMediaRemoteError err)), (commands, origin, replyQ, completion))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingVisibility, void, (MROriginRef origin, MRNowPlayingClientVisibility visibility), (origin, visibility))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteCommandInfoCreate, MRMediaRemoteCommandInfoRef, (CFAllocatorRef allocator), (allocator));
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetCommand, void, (MRMediaRemoteCommandInfoRef commandInfo, MRMediaRemoteCommand command), (commandInfo, command))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetEnabled, void, (MRMediaRemoteCommandInfoRef commandInfo, Boolean enabled), (commandInfo, enabled))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteCommandInfoSetOptions, void, (MRMediaRemoteCommandInfoRef commandInfo, CFDictionaryRef options), (commandInfo, options))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetCanBeNowPlayingApplication, Boolean, (Boolean flag), (flag))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingInfo, void, (CFDictionaryRef info), (info))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingInfoWithMergePolicy, void, (CFDictionaryRef info, MRMediaRemoteMergePolicy mergePolicy), (info, mergePolicy))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetNowPlayingApplicationPlaybackStateForOrigin, void, (MROriginRef origin, MRPlaybackState playbackState, dispatch_queue_t replyQ, void(^completion)(MRMediaRemoteError)), (origin, playbackState, replyQ, completion))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteSetParentApplication, void, (MROriginRef origin, CFStringRef parentAppDisplayID), (origin, parentAppDisplayID))
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoTitle, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtist, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoAlbum, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkData, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkDataHeight, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkDataWidth, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkMIMEType, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoArtworkIdentifier, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoDuration, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoElapsedTime, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoPlaybackRate, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteOptionPlaybackPosition, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteNowPlayingInfoUniqueIdentifier, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteOptionSkipInterval, CFStringRef);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, MediaRemote, kMRMediaRemoteCommandInfoPreferredIntervalsKey, CFStringRef);

#if PLATFORM(IOS_FAMILY)
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaRemote, MRMediaRemoteCopyPickableRoutes, CFArrayRef, (), ());
#endif

#if USE(NOW_PLAYING_ACTIVITY_SUPPRESSION)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, MediaRemote, MRUIControllerProvider);
#endif
