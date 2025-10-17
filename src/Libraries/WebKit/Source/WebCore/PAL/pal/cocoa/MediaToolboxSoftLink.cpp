/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#if USE(MEDIATOOLBOX)

#include <MediaToolbox/MediaToolbox.h>
#include <pal/spi/cocoa/MediaToolboxSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, PAL_EXPORT)

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, FigPhotoDecompressionSetHardwareCutoff, void, (int format, size_t numPixelsCutoff), (format, numPixelsCutoff), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, FigVideoTargetCreateWithVideoReceiverEndpointID, OSStatus, (CFAllocatorRef allocator, xpc_object_t videoReceiverXPCEndpointID, CFDictionaryRef creationOptions, FigVideoTargetRef* videoTargetOut), (allocator, videoReceiverXPCEndpointID, creationOptions, videoTargetOut), PAL_EXPORT)

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, MTShouldPlayHDRVideo, Boolean, (CFArrayRef displayList), (displayList), PAL_EXPORT)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, MTOverrideShouldPlayHDRVideo, void, (Boolean override, Boolean playHDRVideo), (override, playHDRVideo), PAL_EXPORT)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, MT_GetShouldPlayHDRVideoNotificationSingleton, CFTypeRef, (void), (), PAL_EXPORT)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, kMTSupportNotification_ShouldPlayHDRVideoChanged, CFStringRef, PAL_EXPORT)

SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, MediaToolbox, MTAudioProcessingTapGetStorage, void*, (MTAudioProcessingTapRef tap), (tap))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, MediaToolbox, MTAudioProcessingTapGetSourceAudio, OSStatus, (MTAudioProcessingTapRef tap, CMItemCount numberFrames, AudioBufferList *bufferListInOut, MTAudioProcessingTapFlags *flagsOut, CMTimeRange *timeRangeOut, CMItemCount *numberFramesOut), (tap, numberFrames, bufferListInOut, flagsOut, timeRangeOut, numberFramesOut))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, MediaToolbox, MTAudioProcessingTapCreate, OSStatus, (CFAllocatorRef allocator, const MTAudioProcessingTapCallbacks* callbacks, MTAudioProcessingTapCreationFlags flags, MTAudioProcessingTapRef* tapOut), (allocator, callbacks, flags, tapOut), PAL_EXPORT)

#endif // USE(MEDIATOOLBOX)
