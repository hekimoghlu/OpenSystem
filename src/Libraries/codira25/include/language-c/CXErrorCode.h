/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#ifndef LANGUAGE_CORE_C_CXERRORCODE_H
#define LANGUAGE_CORE_C_CXERRORCODE_H

#include "language/Core-c/ExternC.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/**
 * Error codes returned by libclang routines.
 *
 * Zero (\c CXError_Success) is the only error code indicating success.  Other
 * error codes, including not yet assigned non-zero values, indicate errors.
 */
enum CXErrorCode {
  /**
   * No error.
   */
  CXError_Success = 0,

  /**
   * A generic error code, no further details are available.
   *
   * Errors of this kind can get their own specific error codes in future
   * libclang versions.
   */
  CXError_Failure = 1,

  /**
   * libclang crashed while performing the requested operation.
   */
  CXError_Crashed = 2,

  /**
   * The function detected that the arguments violate the function
   * contract.
   */
  CXError_InvalidArguments = 3,

  /**
   * An AST deserialization error has occurred.
   */
  CXError_ASTReadError = 4
};

LANGUAGE_CORE_C_EXTERN_C_END

#endif

