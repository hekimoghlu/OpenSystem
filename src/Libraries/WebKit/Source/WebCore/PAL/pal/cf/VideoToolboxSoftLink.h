/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#if USE(AVFOUNDATION)

#include <VideoToolbox/VTCompressionSession.h>

#include <pal/spi/cf/VideoToolboxSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, VideoToolbox)

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_ExpectedFrameRate, CFStringRef)
#define kVTCompressionPropertyKey_ExpectedFrameRate get_VideoToolbox_kVTCompressionPropertyKey_ExpectedFrameRate()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_MaxKeyFrameInterval, CFStringRef)
#define kVTCompressionPropertyKey_MaxKeyFrameInterval get_VideoToolbox_kVTCompressionPropertyKey_MaxKeyFrameInterval()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, CFStringRef)
#define kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration get_VideoToolbox_kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_RealTime, CFStringRef)
#define kVTCompressionPropertyKey_RealTime get_VideoToolbox_kVTCompressionPropertyKey_RealTime()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_AverageBitRate, CFStringRef)
#define kVTCompressionPropertyKey_AverageBitRate get_VideoToolbox_kVTCompressionPropertyKey_AverageBitRate()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTCompressionPropertyKey_ProfileLevel, CFStringRef)
#define kVTCompressionPropertyKey_ProfileLevel get_VideoToolbox_kVTCompressionPropertyKey_ProfileLevel()

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTEncodeFrameOptionKey_ForceKeyFrame, CFStringRef)
#define kVTEncodeFrameOptionKey_ForceKeyFrame get_VideoToolbox_kVTEncodeFrameOptionKey_ForceKeyFrame()

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder, CFStringRef)
#define kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder get_VideoToolbox_kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTProfileLevel_H264_Baseline_AutoLevel, CFStringRef)
#define kVTProfileLevel_H264_Baseline_AutoLevel get_VideoToolbox_kVTProfileLevel_H264_Baseline_AutoLevel()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTProfileLevel_H264_High_AutoLevel, CFStringRef)
#define kVTProfileLevel_H264_High_AutoLevel get_VideoToolbox_kVTProfileLevel_H264_High_AutoLevel()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, VideoToolbox, kVTProfileLevel_H264_Main_AutoLevel, CFStringRef)
#define kVTProfileLevel_H264_Main_AutoLevel get_VideoToolbox_kVTProfileLevel_H264_Main_AutoLevel()


SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTCompressionSessionCreate, OSStatus, (CFAllocatorRef allocator, int32_t width, int32_t height, CMVideoCodecType codecType, CFDictionaryRef encoderSpecification, CFDictionaryRef sourceImageBufferAttributes, CFAllocatorRef compressedDataAllocator, VTCompressionOutputCallback outputCallback, void* outputCallbackRefCon, VTCompressionSessionRef* compressionSessionOut), (allocator, width, height, codecType, encoderSpecification, sourceImageBufferAttributes, compressedDataAllocator, outputCallback, outputCallbackRefCon, compressionSessionOut))
#define VTCompressionSessionCreate softLink_VideoToolbox_VTCompressionSessionCreate
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTCompressionSessionCompleteFrames, OSStatus, (VTCompressionSessionRef session, CMTime completeUntilPresentationTimeStamp), (session, completeUntilPresentationTimeStamp))
#define VTCompressionSessionCompleteFrames softLink_VideoToolbox_VTCompressionSessionCompleteFrames
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTCompressionSessionEncodeFrame, OSStatus, (VTCompressionSessionRef session, CVImageBufferRef imageBuffer, CMTime presentationTimeStamp, CMTime duration, CFDictionaryRef frameProperties, void* sourceFrameRefcon, VTEncodeInfoFlags* infoFlagsOut), (session, imageBuffer, presentationTimeStamp, duration, frameProperties, sourceFrameRefcon, infoFlagsOut))
#define VTCompressionSessionEncodeFrame softLink_VideoToolbox_VTCompressionSessionEncodeFrame
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTCompressionSessionPrepareToEncodeFrames, OSStatus, (VTCompressionSessionRef session), (session))
#define VTCompressionSessionPrepareToEncodeFrames softLink_VideoToolbox_VTCompressionSessionPrepareToEncodeFrames
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTCompressionSessionInvalidate, void, (VTCompressionSessionRef session), (session))
#define VTCompressionSessionInvalidate softLink_VideoToolbox_VTCompressionSessionInvalidate
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, VideoToolbox, VTGetDefaultColorAttributesWithHints, OSStatus, (OSType codecTypeHint, CFStringRef colorSpaceNameHint, size_t widthHint, size_t heightHint, CFStringRef* colorPrimariesOut, CFStringRef* transferFunctionOut, CFStringRef* yCbCrMatrixOut), (codecTypeHint, colorSpaceNameHint, widthHint, heightHint, colorPrimariesOut, transferFunctionOut, yCbCrMatrixOut))
#define VTGetDefaultColorAttributesWithHints softLink_VideoToolbox_VTGetDefaultColorAttributesWithHints

SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, VideoToolbox, VTRestrictVideoDecoders, OSStatus, (VTVideoDecoderRestrictions restrictionFlags, const CMVideoCodecType* allowedCodecTypeList, CMItemCount allowedCodecTypeCount), (restrictionFlags, allowedCodecTypeList, allowedCodecTypeCount));

#endif // USE(AVFOUNDATION)
