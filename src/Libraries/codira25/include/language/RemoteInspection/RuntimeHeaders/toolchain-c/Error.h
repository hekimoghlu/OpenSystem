/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#ifndef TOOLCHAIN_C_ERROR_H
#define TOOLCHAIN_C_ERROR_H

#include "toolchain-c/ExternC.h"

TOOLCHAIN_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCError Error Handling
 * @ingroup LLVMC
 *
 * @{
 */

#define LLVMErrorSuccess 0

/**
 * Opaque reference to an error instance. Null serves as the 'success' value.
 */
typedef struct LLVMOpaqueError *LLVMErrorRef;

/**
 * Error type identifier.
 */
typedef const void *LLVMErrorTypeId;

/**
 * Returns the type id for the given error instance, which must be a failure
 * value (i.e. non-null).
 */
LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err);

/**
 * Dispose of the given error without handling it. This operation consumes the
 * error, and the given LLVMErrorRef value is not usable once this call returns.
 * Note: This method *only* needs to be called if the error is not being passed
 * to some other consuming operation, e.g. LLVMGetErrorMessage.
 */
void LLVMConsumeError(LLVMErrorRef Err);

/**
 * Returns the given string's error message. This operation consumes the error,
 * and the given LLVMErrorRef value is not usable once this call returns.
 * The caller is responsible for disposing of the string by calling
 * LLVMDisposeErrorMessage.
 */
char *LLVMGetErrorMessage(LLVMErrorRef Err);

/**
 * Dispose of the given error message.
 */
void LLVMDisposeErrorMessage(char *ErrMsg);

/**
 * Returns the type id for toolchain StringError.
 */
LLVMErrorTypeId LLVMGetStringErrorTypeId(void);

/**
 * Create a StringError.
 */
LLVMErrorRef LLVMCreateStringError(const char *ErrMsg);

/**
 * @}
 */

TOOLCHAIN_C_EXTERN_C_END

#endif
