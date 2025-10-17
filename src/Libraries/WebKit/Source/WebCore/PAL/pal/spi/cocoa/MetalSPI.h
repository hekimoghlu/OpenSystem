/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#import <Metal/Metal.h>

#if USE(APPLE_INTERNAL_SDK)

#import <Metal/MTLDevice_Private.h>
#import <Metal/MTLRasterizationRate_Private.h>
#import <Metal/MTLTexture_Private.h>
#import <Metal/MetalPrivate.h>

#else

#import <Foundation/NSObject.h>

typedef struct __IOSurface *IOSurfaceRef;

@protocol MTLDeviceSPI <MTLDevice>
- (NSString*)vendorName;
- (NSString*)familyName;
- (NSString*)productName;
- (id <MTLSharedEvent>)newSharedEventWithMachPort:(mach_port_t)machPort;
@end

@interface _MTLDevice : NSObject
- (void)_purgeDevice;
@end

@protocol MTLRasterizationRateMapDescriptorSPI
@property (nonatomic) float minFactor;
@property (nonatomic) MTLMutability mutability;
@property (nonatomic) BOOL skipSampleValidationAndApplySampleAtTileGranularity;
@end

@interface MTLSharedEventHandle(Private)
- (mach_port_t)eventPort;
@end

#if !PLATFORM(IOS_FAMILY_SIMULATOR)
@interface MTLSharedTextureHandle(Private)
- (instancetype)initWithIOSurface:(IOSurfaceRef)ioSurface label:(NSString*)label;
- (instancetype)initWithMachPort:(mach_port_t)machPort;
@end
#endif

WTF_EXTERN_C_BEGIN

void MTLSetShaderCachePath(NSString *path);

WTF_EXTERN_C_END

#endif
