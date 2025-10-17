/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#ifndef LLVM_C_BITCODEWRITER_H
#define LLVM_C_BITCODEWRITER_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCBitWriter Bit Writer
 * @ingroup LLVMC
 *
 * @{
 */

/*===-- Operations on modules ---------------------------------------------===*/

/** Writes a module to the specified path. Returns 0 on success. */ 
int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char *Path);

/** Writes a module to an open file descriptor. Returns 0 on success. */
int LLVMWriteBitcodeToFD(LLVMModuleRef M, int FD, int ShouldClose,
                         int Unbuffered);

/** Deprecated for LLVMWriteBitcodeToFD. Writes a module to an open file
    descriptor. Returns 0 on success. Closes the Handle. */ 
int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int Handle);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
