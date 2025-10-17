/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
#ifndef LLVM_C_SUPPORT_H
#define LLVM_C_SUPPORT_H

#include <IndexStoreDB_LLVMSupport/toolchain-c_DataTypes.h>
#include <IndexStoreDB_LLVMSupport/toolchain-c_Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function parses the given arguments using the LLVM command line parser.
 * Note that the only stable thing about this function is its signature; you
 * cannot rely on any particular set of command line arguments being interpreted
 * the same way across LLVM versions.
 *
 * @see toolchain::cl::ParseCommandLineOptions()
 */
void LLVMParseCommandLineOptions(int argc, const char *const *argv,
                                 const char *Overview);

#ifdef __cplusplus
}
#endif

#endif
