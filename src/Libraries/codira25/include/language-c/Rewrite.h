/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#ifndef LANGUAGE_CORE_C_REWRITE_H
#define LANGUAGE_CORE_C_REWRITE_H

#include "language/Core-c/CXString.h"
#include "language/Core-c/ExternC.h"
#include "language/Core-c/Index.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

typedef void *CXRewriter;

/**
 * Create CXRewriter.
 */
CINDEX_LINKAGE CXRewriter clang_CXRewriter_create(CXTranslationUnit TU);

/**
 * Insert the specified string at the specified location in the original buffer.
 */
CINDEX_LINKAGE void clang_CXRewriter_insertTextBefore(CXRewriter Rew, CXSourceLocation Loc,
                                           const char *Insert);

/**
 * Replace the specified range of characters in the input with the specified
 * replacement.
 */
CINDEX_LINKAGE void clang_CXRewriter_replaceText(CXRewriter Rew, CXSourceRange ToBeReplaced,
                                      const char *Replacement);

/**
 * Remove the specified range.
 */
CINDEX_LINKAGE void clang_CXRewriter_removeText(CXRewriter Rew, CXSourceRange ToBeRemoved);

/**
 * Save all changed files to disk.
 * Returns 1 if any files were not saved successfully, returns 0 otherwise.
 */
CINDEX_LINKAGE int clang_CXRewriter_overwriteChangedFiles(CXRewriter Rew);

/**
 * Write out rewritten version of the main file to stdout.
 */
CINDEX_LINKAGE void clang_CXRewriter_writeMainFileToStdOut(CXRewriter Rew);

/**
 * Free the given CXRewriter.
 */
CINDEX_LINKAGE void clang_CXRewriter_dispose(CXRewriter Rew);

LANGUAGE_CORE_C_EXTERN_C_END

#endif
