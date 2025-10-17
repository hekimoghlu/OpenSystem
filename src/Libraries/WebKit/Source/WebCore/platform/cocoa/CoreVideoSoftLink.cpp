/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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

#include <CoreVideo/CoreVideo.h>
#include <wtf/SoftLinking.h>

typedef struct __IOSurface* IOSurfaceRef;

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, CoreVideo)

#if HAVE(CVBUFFERCOPYATTACHMENTS)
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVBufferCopyAttachments, CFDictionaryRef, (CVBufferRef buffer, CVAttachmentMode mode), (buffer, mode))
#endif
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVBufferGetAttachment, CFTypeRef, (CVBufferRef buffer, CFStringRef key, CVAttachmentMode* attachmentMode), (buffer, key, attachmentMode))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVBufferGetAttachments, CFDictionaryRef, (CVBufferRef buffer, CVAttachmentMode mode), (buffer, mode))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVBufferRemoveAttachment, void, (CVBufferRef buffer, CFStringRef key), (buffer, key))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVBufferSetAttachment, void, (CVBufferRef buffer, CFStringRef key, CFTypeRef value, CVAttachmentMode attachmentMode), (buffer, key, value, attachmentMode))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetTypeID, CFTypeID, (), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetWidth, size_t, (CVPixelBufferRef pixelBuffer), (pixelBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetWidthOfPlane, size_t, (CVPixelBufferRef pixelBuffer, size_t planeIndex), (pixelBuffer, planeIndex))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetHeight, size_t, (CVPixelBufferRef pixelBuffer), (pixelBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetHeightOfPlane, size_t, (CVPixelBufferRef pixelBuffer, size_t planeIndex), (pixelBuffer, planeIndex))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetBaseAddress, void*, (CVPixelBufferRef pixelBuffer), (pixelBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetBytesPerRow, size_t, (CVPixelBufferRef pixelBuffer), (pixelBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetBytesPerRowOfPlane, size_t, (CVPixelBufferRef pixelBuffer, size_t planeIndex), (pixelBuffer, planeIndex))
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(WebCore, CoreVideo, CVPixelBufferGetPixelFormatType, OSType, (CVPixelBufferRef pixelBuffer), (pixelBuffer), WEBCORE_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetPlaneCount, size_t, (CVPixelBufferRef pixelBuffer), (pixelBuffer))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferGetBaseAddressOfPlane, void *, (CVPixelBufferRef pixelBuffer, size_t planeIndex), (pixelBuffer, planeIndex));
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferLockBaseAddress, CVReturn, (CVPixelBufferRef pixelBuffer, CVOptionFlags lockFlags), (pixelBuffer, lockFlags))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferUnlockBaseAddress, CVReturn, (CVPixelBufferRef pixelBuffer, CVOptionFlags lockFlags), (pixelBuffer, lockFlags))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferPoolCreate, CVReturn,(CFAllocatorRef allocator, CFDictionaryRef poolAttributes, CFDictionaryRef pixelBufferAttributes, CVPixelBufferPoolRef* poolOut), (allocator, poolAttributes, pixelBufferAttributes, poolOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferPoolCreatePixelBuffer, CVReturn, (CFAllocatorRef allocator, CVPixelBufferPoolRef pixelBufferPool, CVPixelBufferRef* pixelBufferOut), (allocator, pixelBufferPool, pixelBufferOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferPoolCreatePixelBufferWithAuxAttributes, CVReturn, (CFAllocatorRef allocator, CVPixelBufferPoolRef pixelBufferPool, CFDictionaryRef auxiliaryAttributes, CVPixelBufferRef* pixelBufferOut), (allocator, pixelBufferPool, auxiliaryAttributes, pixelBufferOut))
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(WebCore, CoreVideo, CVPixelBufferGetIOSurface, IOSurfaceRef, (CVPixelBufferRef pixelBuffer), (pixelBuffer), WEBCORE_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVImageBufferCreateColorSpaceFromAttachments, CGColorSpaceRef, (CFDictionaryRef attachments), (attachments))

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(WebCore, CoreVideo, kCVPixelBufferPixelFormatTypeKey, CFStringRef, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferCGBitmapContextCompatibilityKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferCGImageCompatibilityKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferIOSurfaceOpenGLESFBOCompatibilityKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferWidthKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferHeightKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelFormatOpenGLCompatibility, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelFormatOpenGLESCompatibility, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferIOSurfacePropertiesKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferPoolMinimumBufferCountKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferPoolAllocationThresholdKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrixKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_ITU_R_709_2, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_ITU_R_601_4, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_SMPTE_240M_1995, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferColorPrimaries_EBU_3213, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferColorPrimaries_ITU_R_709_2, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferColorPrimaries_SMPTE_C, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferColorPrimariesKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferTransferFunctionKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferTransferFunction_ITU_R_709_2, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferTransferFunction_SMPTE_240M_1995, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_DCI_P3, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_P3_D65, CFStringRef)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferYCbCrMatrix_ITU_R_2020, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCGColorSpaceKey, CFStringRef)

SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCleanApertureKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCleanApertureHeightKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCleanApertureWidthKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCleanApertureHorizontalOffsetKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVImageBufferCleanApertureVerticalOffsetKey, CFStringRef)

#if PLATFORM(IOS_FAMILY)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferOpenGLESCompatibilityKey, CFStringRef)
#endif

#if PLATFORM(MAC)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferExtendedPixelsRightKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferExtendedPixelsBottomKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreVideo, kCVPixelBufferOpenGLCompatibilityKey, CFStringRef)
#endif

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferCreate, CVReturn, (CFAllocatorRef allocator, size_t width, size_t height, OSType pixelFormatType, CFDictionaryRef pixelBufferAttributes, CVPixelBufferRef *pixelBufferOut), (allocator, width, height, pixelFormatType, pixelBufferAttributes, pixelBufferOut))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CoreVideo, CVPixelBufferCreateWithBytes, CVReturn, (CFAllocatorRef allocator, size_t width, size_t height, OSType pixelFormatType, void* data, size_t bytesPerRow, void (*releaseCallback)(void*, const void*), void* releasePointer, CFDictionaryRef pixelBufferAttributes, CVPixelBufferRef *pixelBufferOut), (allocator, width, height, pixelFormatType, data, bytesPerRow, releaseCallback, releasePointer, pixelBufferAttributes, pixelBufferOut))
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(WebCore, CoreVideo, CVPixelBufferCreateWithIOSurface, CVReturn, (CFAllocatorRef allocator, IOSurfaceRef surface, CFDictionaryRef pixelBufferAttributes, CVPixelBufferRef * pixelBufferOut), (allocator, surface, pixelBufferAttributes, pixelBufferOut), WEBCORE_EXPORT)

