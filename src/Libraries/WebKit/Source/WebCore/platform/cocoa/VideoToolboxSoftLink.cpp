/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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

#include <VideoToolbox/VideoToolbox.h>
#include <wtf/SoftLinking.h>

typedef struct OpaqueVTVideoDecoder VTVideoDecoderRef;
typedef struct OpaqueVTImageRotationSession* VTImageRotationSessionRef;
typedef struct OpaqueVTPixelBufferConformer* VTPixelBufferConformerRef;
typedef struct OpaqueVTPixelTransferSession* VTPixelTransferSessionRef;

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, VideoToolbox)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTSessionCopyProperty, OSStatus, (VTSessionRef session, CFStringRef propertyKey, CFAllocatorRef allocator, void* propertyValueOut), (session, propertyKey, allocator, propertyValueOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTSessionSetProperties, OSStatus, (VTSessionRef session, CFDictionaryRef propertyDictionary), (session, propertyDictionary))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionCreate, OSStatus, (CFAllocatorRef allocator, CMVideoFormatDescriptionRef videoFormatDescription, CFDictionaryRef videoDecoderSpecification, CFDictionaryRef destinationImageBufferAttributes, const VTDecompressionOutputCallbackRecord* outputCallback, VTDecompressionSessionRef* decompressionSessionOut), (allocator, videoFormatDescription, videoDecoderSpecification, destinationImageBufferAttributes, outputCallback, decompressionSessionOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionInvalidate, void, (VTDecompressionSessionRef session), (session))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionCanAcceptFormatDescription, Boolean, (VTDecompressionSessionRef session, CMFormatDescriptionRef newFormatDesc), (session, newFormatDesc))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionWaitForAsynchronousFrames, OSStatus, (VTDecompressionSessionRef session), (session))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionDecodeFrameWithOutputHandler, OSStatus, (VTDecompressionSessionRef session, CMSampleBufferRef sampleBuffer, VTDecodeFrameFlags decodeFlags, VTDecodeInfoFlags *infoFlagsOut, VTDecompressionOutputHandler outputHandler), (session, sampleBuffer, decodeFlags, infoFlagsOut, outputHandler))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTDecompressionSessionDecodeFrame, OSStatus, (VTDecompressionSessionRef session, CMSampleBufferRef sampleBuffer, VTDecodeFrameFlags decodeFlags, void* sourceFrameRefCon, VTDecodeInfoFlags* infoFlagsOut), (session, sampleBuffer, decodeFlags, sourceFrameRefCon, infoFlagsOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTImageRotationSessionCreate, OSStatus, (CFAllocatorRef allocator, uint32_t rotationDegrees, VTImageRotationSessionRef* imageRotationSessionOut), (allocator, rotationDegrees, imageRotationSessionOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTImageRotationSessionSetProperty, OSStatus, (VTImageRotationSessionRef session, CFStringRef propertyKey, CFTypeRef propertyValue), (session, propertyKey, propertyValue))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTImageRotationSessionTransferImage, OSStatus, (VTImageRotationSessionRef session, CVPixelBufferRef sourceBuffer, CVPixelBufferRef destinationBuffer), (session, sourceBuffer, destinationBuffer))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTIsHardwareDecodeSupported, Boolean, (CMVideoCodecType codecType), (codecType))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTGetGVADecoderAvailability, OSStatus, (uint32_t* totalInstanceCountOut, uint32_t* freeInstanceCountOut), (totalInstanceCountOut, freeInstanceCountOut))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTCreateCGImageFromCVPixelBuffer, OSStatus, (CVPixelBufferRef pixelBuffer, CFDictionaryRef options, CGImageRef* imageOut), (pixelBuffer, options, imageOut))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTCopyHEVCDecoderCapabilitiesDictionary, CFDictionaryRef, (), ())
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTGetHEVCCapabilitesForFormatDescription, OSStatus, (CMVideoFormatDescriptionRef formatDescription, CFDictionaryRef decoderCapabilitiesDict, Boolean* isDecodable, Boolean* mayBePlayable), (formatDescription, decoderCapabilitiesDict, isDecodable, mayBePlayable))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTCopyAV1DecoderCapabilitiesDictionary, CFDictionaryRef, (), ())
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, VTRegisterSupplementalVideoDecoderIfAvailable, void, (CMVideoCodecType codecType), (codecType))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTSelectAndCreateVideoDecoderInstance, OSStatus, (CMVideoCodecType codecType, CFAllocatorRef allocator, CFDictionaryRef videoDecoderSpecification, VTVideoDecoderRef *decoderInstanceOut), (codecType, allocator, videoDecoderSpecification, decoderInstanceOut))
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTDecompressionPropertyKey_PixelBufferPool, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTDecompressionPropertyKey_RealTime, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTDecompressionPropertyKey_SuggestedQualityOfServiceTiers, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTImageRotationPropertyKey_EnableHighSpeedTransfer, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTImageRotationPropertyKey_FlipHorizontalOrientation, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTImageRotationPropertyKey_FlipVerticalOrientation, CFStringRef)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTPixelTransferSessionCreate, OSStatus, (CFAllocatorRef allocator, VTPixelTransferSessionRef* pixelTransferSessionOut), (allocator, pixelTransferSessionOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTPixelTransferSessionTransferImage, OSStatus, (VTPixelTransferSessionRef session, CVPixelBufferRef sourceBuffer, CVPixelBufferRef destinationBuffer), (session, sourceBuffer, destinationBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTSessionSetProperty, OSStatus, (VTSessionRef session, CFStringRef propertyKey, CFTypeRef propertyValue), (session, propertyKey, propertyValue))
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTPixelTransferPropertyKey_ScalingMode, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTScalingMode_Letterbox, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTScalingMode_Trim, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTScalingMode_CropSourceToCleanAperture, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTPixelTransferPropertyKey_EnableHardwareAcceleratedTransfer, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTPixelTransferPropertyKey_EnableHighSpeedTransfer, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, VideoToolbox, kVTPixelTransferPropertyKey_RealTime, CFStringRef)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTHEVCDecoderCapability_SupportedProfiles, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTHEVCDecoderCapability_PerProfileSupport, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTHEVCDecoderProfileCapability_IsHardwareAccelerated, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTHEVCDecoderProfileCapability_MaxDecodeLevel, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTHEVCDecoderProfileCapability_MaxPlaybackLevel, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDolbyVisionDecoderCapability_SupportedProfiles, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDolbyVisionDecoderCapability_SupportedLevels, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDolbyVisionDecoderCapability_IsHardwareAccelerated, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderCodecCapability_SupportedProfiles, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderCodecCapability_PerProfileSupport, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderProfileCapability_IsHardwareAccelerated, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderProfileCapability_MaxDecodeLevel, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderProfileCapability_MaxHDRPlaybackLevel, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderProfileCapability_MaxPlaybackLevel, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderCapability_ChromaSubsampling, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, VideoToolbox, kVTDecoderCapability_ColorDepth, CFStringRef)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTPixelBufferConformerCreateWithAttributes, OSStatus, (CFAllocatorRef allocator, CFDictionaryRef attributes, VTPixelBufferConformerRef* conformerOut), (allocator, attributes, conformerOut));
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTPixelBufferConformerIsConformantPixelBuffer, Boolean, (VTPixelBufferConformerRef conformer, CVPixelBufferRef pixBuf), (conformer, pixBuf))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, VideoToolbox, VTPixelBufferConformerCopyConformedPixelBuffer, OSStatus, (VTPixelBufferConformerRef conformer, CVPixelBufferRef sourceBuffer, Boolean ensureModifiable, CVPixelBufferRef* conformedBufferOut), (conformer, sourceBuffer, ensureModifiable, conformedBufferOut))
