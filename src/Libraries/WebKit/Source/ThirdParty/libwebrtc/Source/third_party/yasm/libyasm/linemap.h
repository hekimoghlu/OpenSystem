/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#ifndef YASM_LINEMAP_H
#define YASM_LINEMAP_H

#ifndef YASM_LIB_DECL
#define YASM_LIB_DECL
#endif

/** Create a new line mapping repository.
 * \return New repository.
 */
YASM_LIB_DECL
yasm_linemap *yasm_linemap_create(void);

/** Clean up any memory allocated for a repository.
 * \param linemap       line mapping repository
 */
YASM_LIB_DECL
void yasm_linemap_destroy(yasm_linemap *linemap);

/** Get the current line position in a repository.
 * \param linemap       line mapping repository
 * \return Current virtual line.
 */
YASM_LIB_DECL
unsigned long yasm_linemap_get_current(yasm_linemap *linemap);

/** Get bytecode and source line information, if any, for a virtual line.
 * \param linemap       line mapping repository
 * \param line          virtual line
 * \param bcp           pointer to return bytecode into
 * \param sourcep       pointer to return source code line pointer into
 * \return Zero if source line information available for line, nonzero if not.
 * \note If source line information is not available, bcp and sourcep targets
 * are set to NULL.
 */
YASM_LIB_DECL
int yasm_linemap_get_source(yasm_linemap *linemap, unsigned long line,
                            /*@null@*/ yasm_bytecode **bcp,
                            const char **sourcep);

/** Add bytecode and source line information to the current virtual line.
 * \attention Deletes any existing bytecode and source line information for
 *            the current virtual line.
 * \param linemap       line mapping repository
 * \param bc            bytecode (if any)
 * \param source        source code line
 * \note The source code line pointer is NOT kept, it is strdup'ed.
 */
YASM_LIB_DECL
void yasm_linemap_add_source(yasm_linemap *linemap,
                             /*@null@*/ yasm_bytecode *bc,
                             const char *source);

/** Go to the next line (increments the current virtual line).
 * \param linemap       line mapping repository
 * \return The current (new) virtual line.
 */
YASM_LIB_DECL
unsigned long yasm_linemap_goto_next(yasm_linemap *linemap);

/** Set a new file/line physical association starting point at the specified
 * virtual line.  line_inc indicates how much the "real" line is incremented
 * by for each virtual line increment (0 is perfectly legal).
 * \param linemap       line mapping repository
 * \param filename      physical file name (if NULL, not changed)
 * \param virtual_line  virtual line number (if 0, linemap->current is used)
 * \param file_line     physical line number
 * \param line_inc      line increment
 */
YASM_LIB_DECL
void yasm_linemap_set(yasm_linemap *linemap, /*@null@*/ const char *filename,
                      unsigned long virtual_line, unsigned long file_line,
                      unsigned long line_inc);

/** Poke a single file/line association, restoring the original physical
 * association starting point.  Caution: increments the current virtual line
 * twice.
 * \param linemap       line mapping repository
 * \param filename      physical file name (if NULL, not changed)
 * \param file_line     physical line number
 * \return The virtual line number of the poked association.
 */
YASM_LIB_DECL
unsigned long yasm_linemap_poke(yasm_linemap *linemap,
                                /*@null@*/ const char *filename,
                                unsigned long file_line);

/** Look up the associated physical file and line for a virtual line.
 * \param linemap       line mapping repository
 * \param line          virtual line
 * \param filename      physical file name (output)
 * \param file_line     physical line number (output)
 */
YASM_LIB_DECL
void yasm_linemap_lookup(yasm_linemap *linemap, unsigned long line,
                         /*@out@*/ const char **filename,
                         /*@out@*/ unsigned long *file_line);

/** Traverses all filenames used in a linemap, calling a function on each
 * filename.
 * \param linemap       line mapping repository
 * \param d             data pointer passed to func on each call
 * \param func          function
 * \return Stops early (and returns func's return value) if func returns a
 *         nonzero value; otherwise 0.
 */
YASM_LIB_DECL
int yasm_linemap_traverse_filenames
    (yasm_linemap *linemap, /*@null@*/ void *d,
     int (*func) (const char *filename, void *d));

#endif
