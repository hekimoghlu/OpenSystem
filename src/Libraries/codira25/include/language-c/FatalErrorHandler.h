/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#ifndef LANGUAGE_CORE_C_FATAL_ERROR_HANDLER_H
#define LANGUAGE_CORE_C_FATAL_ERROR_HANDLER_H

#include "language/Core-c/ExternC.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/**
 * Installs error handler that prints error message to stderr and calls abort().
 * Replaces currently installed error handler (if any).
 */
CINDEX_LINKAGE void clang_install_aborting_toolchain_fatal_error_handler(void);

/**
 * Removes currently installed error handler (if any).
 * If no error handler is intalled, the default strategy is to print error
 * message to stderr and call exit(1).
 */
CINDEX_LINKAGE void clang_uninstall_toolchain_fatal_error_handler(void);

LANGUAGE_CORE_C_EXTERN_C_END

#endif
