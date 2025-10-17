/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
//  SymbolicationHelper.h
//  zlog
//
//  Created by Rasha Eqbal on 2/26/18.
//

#ifndef SymbolicationHelper_h
#define SymbolicationHelper_h

#include <CoreFoundation/CoreFoundation.h>
#include <CoreSymbolication/CoreSymbolication.h>

/*
 * Call this function on each address that needs to be symbolicated.
 *
 * sym: The CSSymbolicatorRef which will be used for symbolication. For example, to symbolicate
 *      kernel addresses create a CSSymbolicatorRef by calling CSSymbolicatorCreateWithMachKernel().
 * addr: The address that needs to be symbolicated.
 * binaryImages: The dictionary that aggregates binary image info for offline symbolication.
 */
void PrintSymbolicatedAddress(CSSymbolicatorRef sym, mach_vm_address_t addr, CFMutableDictionaryRef binaryImages);

/*
 * Call this function to dump binary image info required for offline symbolication.
 *
 * binaryImages: The dictionary that stores this info.
 *
 * The preferred way to use this is to create a CFMutableDictionaryRef with a call to CFDictionaryCreateMutable()
 * and pass it in to PrintSymbolicatedAddress() when symbolicating addresses. This will auto-populate the dictionary,
 * which just needs to be passed in here to print the relevant information.
 */
void PrintBinaryImagesInfo(CFMutableDictionaryRef binaryImages);

#endif /* SymbolicationHelper_h */
