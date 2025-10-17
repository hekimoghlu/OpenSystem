/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#import <Metal/MTLCommandBuffer_Private.h>
#import <Metal/MTLDevice_Private.h>
#import <Metal/MTLResource_Private.h>
#import <Metal/MTLTexture_Private.h>
#else
constexpr MTLPixelFormat MTLPixelFormatYCBCR10_420_2P_PACKED = static_cast<MTLPixelFormat>(508);
constexpr MTLPixelFormat MTLPixelFormatYCBCR10_422_2P_PACKED = static_cast<MTLPixelFormat>(509);
constexpr MTLPixelFormat MTLPixelFormatYCBCR10_444_2P_PACKED = static_cast<MTLPixelFormat>(510);

@protocol MTLResourceSPI <MTLResource>
@optional
- (kern_return_t)setOwnerWithIdentity:(mach_port_t)task_id_token;
@end

#if !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(WATCHOS)
@interface MTLSharedTextureHandle(Private)
- (instancetype)initWithMachPort:(mach_port_t)machPort;
@end
#endif

@protocol MTLDeviceSPI <MTLDevice>
- (id <MTLSharedEvent>)newSharedEventWithMachPort:(mach_port_t)machPort;
@end

#endif
