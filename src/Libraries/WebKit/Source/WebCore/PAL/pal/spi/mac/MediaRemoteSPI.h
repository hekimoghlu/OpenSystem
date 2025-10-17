/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#if USE(APPLE_INTERNAL_SDK)

#import <MediaRemote/MediaRemote_Private.h>

#else

enum {
    MRMediaRemoteCommandPlay,
    MRMediaRemoteCommandPause,
    MRMediaRemoteCommandTogglePlayPause,
    MRMediaRemoteCommandStop,
    MRMediaRemoteCommandNextTrack,
    MRMediaRemoteCommandPreviousTrack,
    MRMediaRemoteCommandAdvanceShuffleMode,
    MRMediaRemoteCommandAdvanceRepeatMode,
    MRMediaRemoteCommandBeginFastForward,
    MRMediaRemoteCommandEndFastForward,
    MRMediaRemoteCommandBeginRewind,
    MRMediaRemoteCommandEndRewind,
    MRMediaRemoteCommandRewind15Seconds,
    MRMediaRemoteCommandFastForward15Seconds,
    MRMediaRemoteCommandRewind30Seconds,
    MRMediaRemoteCommandFastForward30Seconds,
    MRMediaRemoteCommandToggleRecord,
    MRMediaRemoteCommandSkipForward,
    MRMediaRemoteCommandSkipBackward,
    MRMediaRemoteCommandChangePlaybackRate,
    MRMediaRemoteCommandRateTrack,
    MRMediaRemoteCommandLikeTrack,
    MRMediaRemoteCommandDislikeTrack,
    MRMediaRemoteCommandBookmarkTrack,
    MRMediaRemoteCommandSeekToPlaybackPosition,
    MRMediaRemoteCommandChangeRepeatMode,
    MRMediaRemoteCommandChangeShuffleMode,
    MRMediaRemoteCommandEnableLanguageOption,
    MRMediaRemoteCommandDisableLanguageOption
};
typedef uint32_t MRMediaRemoteCommand;

enum {
    kMRPlaybackStateUnknown = 0,
    kMRPlaybackStatePlaying,
    kMRPlaybackStatePaused,
    kMRPlaybackStateStopped,
    kMRPlaybackStateInterrupted
};
typedef uint32_t MRPlaybackState;

enum {
    MRMediaRemoteCommandHandlerStatusSuccess = 0,
    MRMediaRemoteCommandHandlerStatusNoSuchContent = 1,
    MRMediaRemoteCommandHandlerStatusCommandFailed = 2,
    MRMediaRemoteCommandHandlerStatusNoActionableNowPlayingItem = 10,
    MRMediaRemoteCommandHandlerStatusUIKitLegacy = 3
};
typedef uint32_t MRMediaRemoteCommandHandlerStatus;

enum {
    MRNowPlayingClientVisibilityUndefined = 0,
    MRNowPlayingClientVisibilityAlwaysVisible,
    MRNowPlayingClientVisibilityVisibleWhenBackgrounded,
    MRNowPlayingClientVisibilityNeverVisible
};
typedef uint32_t MRNowPlayingClientVisibility;

enum : uint8_t {
    MRMediaRemoteMergePolicyUpdate = 0,
    MRMediaRemoteMergePolicyReplace,
};
typedef uint8_t MRMediaRemoteMergePolicy;

typedef uint32_t MRMediaRemoteError;
typedef uint32_t MRSendCommandAppOptions;
typedef uint32_t MRSendCommandError;
typedef struct _MROrigin *MROriginRef;
typedef struct _MRMediaRemoteCommandInfo *MRMediaRemoteCommandInfoRef;
typedef void *MRNowPlayingClientRef;
typedef void(^MRMediaRemoteAsyncCommandHandlerBlock)(MRMediaRemoteCommand command, CFDictionaryRef options, void(^completion)(CFArrayRef));

WTF_EXTERN_C_BEGIN

#pragma mark - MRRemoteControl

void* MRMediaRemoteAddAsyncCommandHandlerBlock(MRMediaRemoteAsyncCommandHandlerBlock);
void MRMediaRemoteRemoveCommandHandlerBlock(void *observer);
void MRMediaRemoteSetSupportedCommands(CFArrayRef, MROriginRef, dispatch_queue_t, void(^completion)(MRMediaRemoteError));
void MRMediaRemoteGetSupportedCommandsForOrigin(MROriginRef, dispatch_queue_t, void(^completion)(CFArrayRef));
void MRMediaRemoteSetNowPlayingVisibility(MROriginRef, MRNowPlayingClientVisibility);
Boolean MRMediaRemoteSendCommandToApp(MRMediaRemoteCommand, CFDictionaryRef, MROriginRef, CFStringRef, MRSendCommandAppOptions, dispatch_queue_t, void(^completion)(MRSendCommandError, CFArrayRef));

#pragma mark - MROrigin

MROriginRef MRMediaRemoteGetLocalOrigin();

#pragma mark - MRCommandInfo

MRMediaRemoteCommandInfoRef MRMediaRemoteCommandInfoCreate(CFAllocatorRef);
void MRMediaRemoteCommandInfoSetCommand(MRMediaRemoteCommandInfoRef, MRMediaRemoteCommand);
void MRMediaRemoteCommandInfoSetEnabled(MRMediaRemoteCommandInfoRef, Boolean);
void MRMediaRemoteCommandInfoSetOptions(MRMediaRemoteCommandInfoRef, CFDictionaryRef);

#pragma mark - MRNowPlaying

Boolean MRMediaRemoteSetCanBeNowPlayingApplication(Boolean);
void MRMediaRemoteSetNowPlayingApplicationPlaybackStateForOrigin(MROriginRef, MRPlaybackState, dispatch_queue_t replyQ, void(^completion)(MRMediaRemoteError));
void MRMediaRemoteSetNowPlayingInfo(CFDictionaryRef);
void MRMediaRemoteSetNowPlayingInfoWithMergePolicy(CFDictionaryRef, MRMediaRemoteMergePolicy);

#pragma mark - MRAVRouting

CFArrayRef MRMediaRemoteCopyPickableRoutes();

WTF_EXTERN_C_END

@protocol MRUIControllable <NSObject>
@end

@protocol MRNowPlayingActivityUIControllable <MRUIControllable>
@end

@interface MRUIControllerProvider : NSObject
+ (id<MRNowPlayingActivityUIControllable>)nowPlayingActivityController;
@end

#endif // USE(APPLE_INTERNAL_SDK)
