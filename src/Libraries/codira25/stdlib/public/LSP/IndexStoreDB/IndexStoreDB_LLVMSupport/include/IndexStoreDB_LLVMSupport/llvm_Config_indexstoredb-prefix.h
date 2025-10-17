/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef INDEXSTOREDB_PREFIX_H
#define INDEXSTOREDB_PREFIX_H

/* HACK: Rename all of the toolchain symbols so that they will not collide if another
 * copy of toolchain is linked into the same image. The use of toolchain within IndexStore
 * is purely an implementation detail. Using a source-level rename is a
 * workaround for the lack of symbol visibility controls in languagepm. Ideally we
 * could do this with a combination of `-fvisibility=hidden` and `ld -r`.
*/

#define toolchain indexstoredb_toolchain
#define LLVMEnablePrettyStackTrace indexstoredb_LLVMEnablePrettyStackTrace
#define LLVMParseCommandLineOptions indexstoredb_LLVMParseCommandLineOptions
#define LLVMResetFatalErrorHandler indexstoredb_LLVMResetFatalErrorHandler
#define LLVMInstallFatalErrorHandler indexstoredb_LLVMInstallFatalErrorHandler

#endif // INDEXSTOREDB_PREFIX_H
