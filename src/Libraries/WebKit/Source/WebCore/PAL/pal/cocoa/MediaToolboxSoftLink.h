/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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

#if USE(MEDIATOOLBOX)

#include <pal/spi/cocoa/MediaToolboxSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, MediaToolbox)

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, FigPhotoDecompressionSetHardwareCutoff, void, (int, size_t numPixelsCutoff), (format, numPixelsCutoff))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, MediaToolbox, FigVideoTargetCreateWithVideoReceiverEndpointID, OSStatus, (CFAllocatorRef allocator, xpc_object_t videoReceiverXPCEndpointID, CFDictionaryRef creationOptions, FigVideoTargetRef* videoTargetOut), (allocator, videoReceiverXPCEndpointID, creationOptions, videoTargetOut))
#define FigVideoTargetCreateWithVideoReceiverEndpointID PAL::softLink_MediaToolbox_FigVideoTargetCreateWithVideoReceiverEndpointID

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, MTShouldPlayHDRVideo, Boolean, (CFArrayRef displayList), (displayList))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, MTOverrideShouldPlayHDRVideo, void, (Boolean override, Boolean playHDRVideo), (override, playHDRVideo))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, MT_GetShouldPlayHDRVideoNotificationSingleton, CFTypeRef, (void), ())
#define MT_GetShouldPlayHDRVideoNotificationSingleton PAL::softLinkMediaToolboxMT_GetShouldPlayHDRVideoNotificationSingleton

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, kMTSupportNotification_ShouldPlayHDRVideoChanged, CFStringRef)
#define kMTSupportNotification_ShouldPlayHDRVideoChanged PAL::get_MediaToolbox_kMTSupportNotification_ShouldPlayHDRVideoChanged()

SOFT_LINK_FUNCTION_FOR_HEADER(PAL, MediaToolbox, MTAudioProcessingTapGetStorage, void*, (MTAudioProcessingTapRef tap), (tap))
#define MTAudioProcessingTapGetStorage softLink_MediaToolbox_MTAudioProcessingTapGetStorage
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, MediaToolbox, MTAudioProcessingTapGetSourceAudio, OSStatus, (MTAudioProcessingTapRef tap, CMItemCount numberFrames, AudioBufferList *bufferListInOut, MTAudioProcessingTapFlags *flagsOut, CMTimeRange *timeRangeOut, CMItemCount *numberFramesOut), (tap, numberFrames, bufferListInOut, flagsOut, timeRangeOut, numberFramesOut))
#define MTAudioProcessingTapGetSourceAudio softLink_MediaToolbox_MTAudioProcessingTapGetSourceAudio
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, MediaToolbox, MTAudioProcessingTapCreate, OSStatus, (CFAllocatorRef allocator, const MTAudioProcessingTapCallbacks* callbacks, MTAudioProcessingTapCreationFlags flags, MTAudioProcessingTapRef* tapOut), (allocator, callbacks, flags, tapOut))
#define MTAudioProcessingTapCreate softLink_MediaToolbox_MTAudioProcessingTapCreate

#endif // USE(MEDIATOOLBOX)
