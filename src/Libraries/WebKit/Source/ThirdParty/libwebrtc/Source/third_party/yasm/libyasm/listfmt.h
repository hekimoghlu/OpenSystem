/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#ifndef YASM_LISTFMT_H
#define YASM_LISTFMT_H

#ifndef YASM_DOXYGEN
/** Base #yasm_listfmt structure.  Must be present as the first element in any
 * #yasm_listfmt implementation.
 */
typedef struct yasm_listfmt_base {
    /** #yasm_listfmt_module implementation for this list format. */
    const struct yasm_listfmt_module *module;
} yasm_listfmt_base;
#endif

/** YASM list format module interface. */
typedef struct yasm_listfmt_module {
    /** One-line description of the list format. */
    const char *name;

    /** Keyword used to select list format. */
    const char *keyword;

    /** Create list format.
     * Module-level implementation of yasm_listfmt_create().
     * The filenames are provided solely for informational purposes.
     * \param in_filename   primary input filename
     * \param obj_filename  object filename
     * \return NULL if unable to initialize.
     */
    /*@null@*/ /*@only@*/ yasm_listfmt * (*create)
        (const char *in_filename, const char *obj_filename);

    /** Module-level implementation of yasm_listfmt_destroy().
     * Call yasm_listfmt_destroy() instead of calling this function.
     */
    void (*destroy) (/*@only@*/ yasm_listfmt *listfmt);

    /** Module-level implementation of yasm_listfmt_output().
     * Call yasm_listfmt_output() instead of calling this function.
     */
    void (*output) (yasm_listfmt *listfmt, FILE *f, yasm_linemap *linemap,
                    yasm_arch *arch);
} yasm_listfmt_module;

/** Get the keyword used to select a list format.
 * \param listfmt   list format
 * \return keyword
 */
const char *yasm_listfmt_keyword(const yasm_listfmt *listfmt);

/** Initialize list format for use.  Must call before any other list
 * format functions.  The filenames are provided solely for informational
 * purposes.
 * \param module        list format module
 * \param in_filename   primary input filename
 * \param obj_filename  object filename
 * \return NULL if object format does not provide needed support.
 */
/*@null@*/ /*@only@*/ yasm_listfmt *yasm_listfmt_create
    (const yasm_listfmt_module *module, const char *in_filename,
     const char *obj_filename);

/** Cleans up any allocated list format memory.
 * \param listfmt       list format
 */
void yasm_listfmt_destroy(/*@only@*/ yasm_listfmt *listfmt);

/** Write out list to the list file.
 * This function may call all read-only yasm_* functions as necessary.
 * \param listfmt       list format
 * \param f             output list file
 * \param linemap       line mapping repository
 * \param arch          architecture
 */
void yasm_listfmt_output(yasm_listfmt *listfmt, FILE *f,
                         yasm_linemap *linemap, yasm_arch *arch);

#ifndef YASM_DOXYGEN

/* Inline macro implementations for listfmt functions */

#define yasm_listfmt_keyword(listfmt) \
    (((yasm_listfmt_base *)listfmt)->module->keyword)

#define yasm_listfmt_create(module, in_filename, obj_filename) \
    module->create(in_filename, obj_filename)

#define yasm_listfmt_destroy(listfmt) \
    ((yasm_listfmt_base *)listfmt)->module->destroy(listfmt)

#define yasm_listfmt_output(listfmt, f, linemap, a) \
    ((yasm_listfmt_base *)listfmt)->module->output(listfmt, f, linemap, a)

#endif

#endif
