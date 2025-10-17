/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

#import <wtf/SoftLinking.h>

#if HAVE(MEDIA_USAGE_FRAMEWORK)

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, PAL_EXPORT);

SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(PAL, UsageTracking, USVideoUsage, PAL_EXPORT);

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyCanShowControlsManager, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyCanShowNowPlayingControls, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsSuspended, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsInActiveDocument, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsFullscreen, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsMuted, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsMediaDocumentInMainFrame, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsAudio, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyAudioElementWithUserGesture, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyUserHasPlayedAudioBefore, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsElementRectMostlyInMainFrame, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyNoAudio, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyPlaybackPermitted, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyPageMediaPlaybackSuspended, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyNoUserGestureRequired, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyHasEverNotifiedAboutPlaying, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyOutsideOfFullscreen, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsVideo, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyRenderer, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyNoVideo, NSString*, PAL_EXPORT);
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, UsageTracking, USVideoMetadataKeyIsLargeEnoughForMainContent, NSString*, PAL_EXPORT);

#endif // HAVE(MEDIA_USAGE_FRAMEWORK)
