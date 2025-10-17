/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

#if USE(AVFOUNDATION)

#include <VideoToolbox/VTCompressionSession.h>

#include <pal/spi/cf/VideoToolboxSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, VideoToolbox, PAL_EXPORT)

SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_ExpectedFrameRate, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_MaxKeyFrameInterval, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_RealTime, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_AverageBitRate, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTCompressionPropertyKey_ProfileLevel, CFStringRef)

SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTEncodeFrameOptionKey_ForceKeyFrame, CFStringRef)

SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTProfileLevel_H264_Baseline_AutoLevel, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTProfileLevel_H264_High_AutoLevel, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, VideoToolbox, kVTProfileLevel_H264_Main_AutoLevel, CFStringRef)

SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTCompressionSessionCreate, OSStatus, (CFAllocatorRef allocator, int32_t width, int32_t height, CMVideoCodecType codecType, CFDictionaryRef encoderSpecification, CFDictionaryRef sourceImageBufferAttributes, CFAllocatorRef compressedDataAllocator, VTCompressionOutputCallback outputCallback, void* outputCallbackRefCon, VTCompressionSessionRef* compressionSessionOut), (allocator, width, height, codecType, encoderSpecification, sourceImageBufferAttributes, compressedDataAllocator, outputCallback, outputCallbackRefCon, compressionSessionOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTCompressionSessionCompleteFrames, OSStatus, (VTCompressionSessionRef session, CMTime completeUntilPresentationTimeStamp), (session, completeUntilPresentationTimeStamp))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTCompressionSessionEncodeFrame, OSStatus, (VTCompressionSessionRef session, CVImageBufferRef imageBuffer, CMTime presentationTimeStamp, CMTime duration, CFDictionaryRef frameProperties, void* sourceFrameRefcon, VTEncodeInfoFlags* infoFlagsOut), (session, imageBuffer, presentationTimeStamp, duration, frameProperties, sourceFrameRefcon, infoFlagsOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTCompressionSessionPrepareToEncodeFrames, OSStatus, (VTCompressionSessionRef session), (session))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTCompressionSessionInvalidate, void, (VTCompressionSessionRef session), (session))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, VideoToolbox, VTGetDefaultColorAttributesWithHints, OSStatus, (OSType codecTypeHint, CFStringRef colorSpaceNameHint, size_t widthHint, size_t heightHint, CFStringRef* colorPrimariesOut, CFStringRef* transferFunctionOut, CFStringRef* yCbCrMatrixOut), (codecTypeHint, colorSpaceNameHint, widthHint, heightHint, colorPrimariesOut, transferFunctionOut, yCbCrMatrixOut))

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, VideoToolbox, VTRestrictVideoDecoders, OSStatus, (VTVideoDecoderRestrictions restrictionFlags, const CMVideoCodecType* allowedCodecTypeList, CMItemCount allowedCodecTypeCount), (restrictionFlags, allowedCodecTypeList, allowedCodecTypeCount), PAL_EXPORT);

#endif // USE(AVFOUNDATION)
