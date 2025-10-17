/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

#if HAVE(MEDIA_USAGE_FRAMEWORK)

#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, UsageTracking);

SOFT_LINK_CLASS_FOR_HEADER(PAL, USVideoUsage);
#define _AXSIsolatedTreeModeFunctionIsAvailable PAL::canLoad_libAccessibility__AXSIsolatedTreeMode

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyCanShowControlsManager, NSString *)
#define USVideoMetadataKeyCanShowControlsManager PAL::get_UsageTracking_USVideoMetadataKeyCanShowControlsManager()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyCanShowNowPlayingControls, NSString *)
#define USVideoMetadataKeyCanShowNowPlayingControls PAL::get_UsageTracking_USVideoMetadataKeyCanShowNowPlayingControls()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsSuspended, NSString *)
#define USVideoMetadataKeyIsSuspended PAL::get_UsageTracking_USVideoMetadataKeyIsSuspended()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsInActiveDocument, NSString *)
#define USVideoMetadataKeyIsInActiveDocument PAL::get_UsageTracking_USVideoMetadataKeyIsInActiveDocument()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsFullscreen, NSString *)
#define USVideoMetadataKeyIsFullscreen PAL::get_UsageTracking_USVideoMetadataKeyIsFullscreen()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsMuted, NSString *)
#define USVideoMetadataKeyIsMuted PAL::get_UsageTracking_USVideoMetadataKeyIsMuted()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsMediaDocumentInMainFrame, NSString *)
#define USVideoMetadataKeyIsMediaDocumentInMainFrame PAL::get_UsageTracking_USVideoMetadataKeyIsMediaDocumentInMainFrame()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsAudio, NSString *)
#define USVideoMetadataKeyIsAudio PAL::get_UsageTracking_USVideoMetadataKeyIsAudio()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyAudioElementWithUserGesture, NSString *)
#define USVideoMetadataKeyAudioElementWithUserGesture PAL::get_UsageTracking_USVideoMetadataKeyAudioElementWithUserGesture()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyUserHasPlayedAudioBefore, NSString *)
#define USVideoMetadataKeyUserHasPlayedAudioBefore PAL::get_UsageTracking_USVideoMetadataKeyUserHasPlayedAudioBefore()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsElementRectMostlyInMainFrame, NSString *)
#define USVideoMetadataKeyIsElementRectMostlyInMainFrame PAL::get_UsageTracking_USVideoMetadataKeyIsElementRectMostlyInMainFrame()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyNoAudio, NSString *)
#define USVideoMetadataKeyNoAudio PAL::get_UsageTracking_USVideoMetadataKeyNoAudio()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyPlaybackPermitted, NSString *)
#define USVideoMetadataKeyPlaybackPermitted PAL::get_UsageTracking_USVideoMetadataKeyPlaybackPermitted()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyPageMediaPlaybackSuspended, NSString *)
#define USVideoMetadataKeyPageMediaPlaybackSuspended PAL::get_UsageTracking_USVideoMetadataKeyPageMediaPlaybackSuspended()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement, NSString *)
#define USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement PAL::get_UsageTracking_USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline, NSString *)
#define USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline PAL::get_UsageTracking_USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted, NSString *)
#define USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted PAL::get_UsageTracking_USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange, NSString *)
#define USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange PAL::get_UsageTracking_USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange, NSString *)
#define USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange PAL::get_UsageTracking_USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode, NSString *)
#define USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode PAL::get_UsageTracking_USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyNoUserGestureRequired, NSString *)
#define USVideoMetadataKeyNoUserGestureRequired PAL::get_UsageTracking_USVideoMetadataKeyNoUserGestureRequired()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying, NSString *)
#define USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying PAL::get_UsageTracking_USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyHasEverNotifiedAboutPlaying, NSString *)
#define USVideoMetadataKeyHasEverNotifiedAboutPlaying PAL::get_UsageTracking_USVideoMetadataKeyHasEverNotifiedAboutPlaying()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyOutsideOfFullscreen, NSString *)
#define USVideoMetadataKeyOutsideOfFullscreen PAL::get_UsageTracking_USVideoMetadataKeyOutsideOfFullscreen()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsVideo, NSString *)
#define USVideoMetadataKeyIsVideo PAL::get_UsageTracking_USVideoMetadataKeyIsVideo()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyRenderer, NSString *)
#define USVideoMetadataKeyRenderer PAL::get_UsageTracking_USVideoMetadataKeyRenderer()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyNoVideo, NSString *)
#define USVideoMetadataKeyNoVideo PAL::get_UsageTracking_USVideoMetadataKeyNoVideo()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, UsageTracking, USVideoMetadataKeyIsLargeEnoughForMainContent, NSString *)
#define USVideoMetadataKeyIsLargeEnoughForMainContent PAL::get_UsageTracking_USVideoMetadataKeyIsLargeEnoughForMainContent()

#endif // HAVE(MEDIA_USAGE_FRAMEWORK)
