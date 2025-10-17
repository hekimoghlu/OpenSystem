/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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
#ifndef LANGUAGE_CORE_C_BUILDSYSTEM_H
#define LANGUAGE_CORE_C_BUILDSYSTEM_H

#include "language/Core-c/CXErrorCode.h"
#include "language/Core-c/CXString.h"
#include "language/Core-c/ExternC.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/**
 * \defgroup BUILD_SYSTEM Build system utilities
 * @{
 */

/**
 * Return the timestamp for use with Clang's
 * \c -fbuild-session-timestamp= option.
 */
CINDEX_LINKAGE unsigned long long clang_getBuildSessionTimestamp(void);

/**
 * Object encapsulating information about overlaying virtual
 * file/directories over the real file system.
 */
typedef struct CXVirtualFileOverlayImpl *CXVirtualFileOverlay;

/**
 * Create a \c CXVirtualFileOverlay object.
 * Must be disposed with \c clang_VirtualFileOverlay_dispose().
 *
 * \param options is reserved, always pass 0.
 */
CINDEX_LINKAGE CXVirtualFileOverlay
clang_VirtualFileOverlay_create(unsigned options);

/**
 * Map an absolute virtual file path to an absolute real one.
 * The virtual path must be canonicalized (not contain "."/"..").
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_addFileMapping(CXVirtualFileOverlay,
                                        const char *virtualPath,
                                        const char *realPath);

/**
 * Set the case sensitivity for the \c CXVirtualFileOverlay object.
 * The \c CXVirtualFileOverlay object is case-sensitive by default, this
 * option can be used to override the default.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_setCaseSensitivity(CXVirtualFileOverlay,
                                            int caseSensitive);

/**
 * Write out the \c CXVirtualFileOverlay object to a char buffer.
 *
 * \param options is reserved, always pass 0.
 * \param out_buffer_ptr pointer to receive the buffer pointer, which should be
 * disposed using \c clang_free().
 * \param out_buffer_size pointer to receive the buffer size.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_writeToBuffer(CXVirtualFileOverlay, unsigned options,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size);

/**
 * free memory allocated by libclang, such as the buffer returned by
 * \c CXVirtualFileOverlay() or \c clang_ModuleMapDescriptor_writeToBuffer().
 *
 * \param buffer memory pointer to free.
 */
CINDEX_LINKAGE void clang_free(void *buffer);

/**
 * Dispose a \c CXVirtualFileOverlay object.
 */
CINDEX_LINKAGE void clang_VirtualFileOverlay_dispose(CXVirtualFileOverlay);

/**
 * Object encapsulating information about a module.modulemap file.
 */
typedef struct CXModuleMapDescriptorImpl *CXModuleMapDescriptor;

/**
 * Create a \c CXModuleMapDescriptor object.
 * Must be disposed with \c clang_ModuleMapDescriptor_dispose().
 *
 * \param options is reserved, always pass 0.
 */
CINDEX_LINKAGE CXModuleMapDescriptor
clang_ModuleMapDescriptor_create(unsigned options);

/**
 * Sets the framework module name that the module.modulemap describes.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_setFrameworkModuleName(CXModuleMapDescriptor,
                                                 const char *name);

/**
 * Sets the umbrella header name that the module.modulemap describes.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_setUmbrellaHeader(CXModuleMapDescriptor,
                                            const char *name);

/**
 * Write out the \c CXModuleMapDescriptor object to a char buffer.
 *
 * \param options is reserved, always pass 0.
 * \param out_buffer_ptr pointer to receive the buffer pointer, which should be
 * disposed using \c clang_free().
 * \param out_buffer_size pointer to receive the buffer size.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_writeToBuffer(CXModuleMapDescriptor, unsigned options,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size);

/**
 * Dispose a \c CXModuleMapDescriptor object.
 */
CINDEX_LINKAGE void clang_ModuleMapDescriptor_dispose(CXModuleMapDescriptor);

/**
 * @}
 */

LANGUAGE_CORE_C_EXTERN_C_END

#endif /* CLANG_C_BUILD_SYSTEM_H */

