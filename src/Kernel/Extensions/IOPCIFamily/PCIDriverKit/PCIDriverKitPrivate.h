/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

//
//  PCIDriverKitPrivate.h
//  PCIDriverKit
//
//  Created by Kevin Strasberg on 10/23/19.
//

#ifndef PCIDriverKitPrivate_h
#define PCIDriverKitPrivate_h

enum kPCIDriverKitMemoryAccessOperation
{
    kPCIDriverKitMemoryAccessOperationDeviceMemoryIndexMask = 0x000000FF,
    kPCIDriverKitMemoryAccessOperationAccessTypeMask        = 0x0000FF00,
    kPCIDriverKitMemoryAccessOperationDeviceRead            = 0x00000100,
    kPCIDriverKitMemoryAccessOperationDeviceWrite           = 0x00000200,
    kPCIDriverKitMemoryAccessOperationConfigurationRead     = 0x00000400,
    kPCIDriverKitMemoryAccessOperationConfigurationWrite    = 0x00000800,
    kPCIDriverKitMemoryAccessOperationIORead                = 0x00001000,
    kPCIDriverKitMemoryAccessOperationIOWrite               = 0x00002000,

    kPCIDriverKitMemoryAccessOperationSizeMask = 0x000F0000,
    kPCIDriverKitMemoryAccessOperation8Bit     = 0x00010000,
    kPCIDriverKitMemoryAccessOperation16Bit    = 0x00020000,
    kPCIDriverKitMemoryAccessOperation32Bit    = 0x00040000,
    kPCIDriverKitMemoryAccessOperation64Bit    = 0x00080000
};

#endif /* PCIDriverKitPrivate_h */
