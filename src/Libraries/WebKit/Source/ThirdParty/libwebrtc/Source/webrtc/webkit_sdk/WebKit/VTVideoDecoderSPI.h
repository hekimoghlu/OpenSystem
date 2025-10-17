/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include <VideoToolbox/VideoToolbox.h>
#include "CMBaseObjectSPI.h"

#if defined __has_include && __has_include(<CoreFoundation/CFPriv.h>)
#include <VideoToolbox/VTVideoDecoder.h>
#include <VideoToolbox/VTVideoDecoderRegistration.h>
#else

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 4)

typedef FourCharCode FigVideoCodecType;
typedef struct OpaqueVTVideoDecoder* VTVideoDecoderRef;
typedef struct OpaqueVTVideoDecoderSession* VTVideoDecoderSession;
typedef struct OpaqueVTVideoDecoderFrame* VTVideoDecoderFrame;

typedef OSStatus (*VTVideoDecoderFunction_CreateInstance)(FourCharCode codecType, CFAllocatorRef allocator, VTVideoDecoderRef *instanceOut);
typedef OSStatus (*VTVideoDecoderFunction_StartSession)(VTVideoDecoderRef, VTVideoDecoderSession, CMVideoFormatDescriptionRef);
typedef OSStatus (*VTVideoDecoderFunction_DecodeFrame)(VTVideoDecoderRef, VTVideoDecoderFrame, CMSampleBufferRef, VTDecodeFrameFlags, VTDecodeInfoFlags*);
typedef OSStatus (*VTVideoDecoderFunction_TBD)();

enum {
    kVTVideoDecoder_ClassVersion_1 = 1,
    kVTVideoDecoder_ClassVersion_2 = 2,
    kVTVideoDecoder_ClassVersion_3 = 3,
};

typedef struct {
    CMBaseClassVersion version;

    VTVideoDecoderFunction_StartSession startSession;
    VTVideoDecoderFunction_DecodeFrame decodeFrame;
    
    VTVideoDecoderFunction_TBD copySupportedPropertyDictionary;
    VTVideoDecoderFunction_TBD setProperties;

    VTVideoDecoderFunction_TBD copySerializableProperties;
    VTVideoDecoderFunction_TBD canAcceptFormatDescription;
    VTVideoDecoderFunction_TBD finishDelayedFrames;
    VTVideoDecoderFunction_TBD reserved7;
    VTVideoDecoderFunction_TBD reserved8;
    VTVideoDecoderFunction_TBD reserved9;
} VTVideoDecoderClass;

typedef struct {
    CMBaseVTable base;
    const VTVideoDecoderClass *videoDecoderClass;
} VTVideoDecoderVTable;

VT_EXPORT CMBaseClassID VTVideoDecoderGetClassID(void);
VT_EXPORT CVPixelBufferPoolRef VTDecoderSessionGetPixelBufferPool(VTVideoDecoderSession session );
VT_EXPORT OSStatus VTDecoderSessionSetPixelBufferAttributes(VTVideoDecoderSession session, CFDictionaryRef decompressorPixelBufferAttributes);
VT_EXPORT OSStatus VTDecoderSessionEmitDecodedFrame(VTVideoDecoderSession session, VTVideoDecoderFrame frame, OSStatus status, VTDecodeInfoFlags infoFlags, CVImageBufferRef imageBuffer);
VT_EXPORT OSStatus VTRegisterVideoDecoder(FigVideoCodecType, VTVideoDecoderFunction_CreateInstance);

#pragma pack(pop)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __has_include && __has_include(<CoreFoundation/CFPriv.h>)
