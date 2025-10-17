/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#ifndef YASM_DBGFMT_H
#define YASM_DBGFMT_H

#ifndef YASM_DOXYGEN
/** Base #yasm_dbgfmt structure.  Must be present as the first element in any
 * #yasm_dbgfmt implementation.
 */
typedef struct yasm_dbgfmt_base {
    /** #yasm_dbgfmt_module implementation for this debug format. */
    const struct yasm_dbgfmt_module *module;
} yasm_dbgfmt_base;
#endif

/** Debug format module interface. */
struct yasm_dbgfmt_module {
    /** One-line description of the debug format. */
    const char *name;

    /** Keyword used to select debug format. */
    const char *keyword;

    /** NULL-terminated list of directives.  NULL if none. */
    /*@null@*/ const yasm_directive *directives;

    /** Create debug format.
     * Module-level implementation of yasm_dbgfmt_create().
     * The filenames are provided solely for informational purposes.
     * \param object        object
     * \return NULL if object format does not provide needed support.
     */
    /*@null@*/ /*@only@*/ yasm_dbgfmt * (*create) (yasm_object *object);

    /** Module-level implementation of yasm_dbgfmt_destroy().
     * Call yasm_dbgfmt_destroy() instead of calling this function.
     */
    void (*destroy) (/*@only@*/ yasm_dbgfmt *dbgfmt);

    /** Module-level implementation of yasm_dbgfmt_generate().
     * Call yasm_dbgfmt_generate() instead of calling this function.
     */
    void (*generate) (yasm_object *object, yasm_linemap *linemap,
                      yasm_errwarns *errwarns);
};

/** Get the keyword used to select a debug format.
 * \param dbgfmt    debug format
 * \return keyword
 */
const char *yasm_dbgfmt_keyword(const yasm_dbgfmt *dbgfmt);

/** Initialize debug output for use.  Must call before any other debug
 * format functions.  The filenames are provided solely for informational
 * purposes.
 * \param module        debug format module
 * \param object        object to generate debugging information for
 * \return NULL if object format does not provide needed support.
 */
/*@null@*/ /*@only@*/ yasm_dbgfmt *yasm_dbgfmt_create
    (const yasm_dbgfmt_module *module, yasm_object *object);

/** Cleans up any allocated debug format memory.
 * \param dbgfmt        debug format
 */
void yasm_dbgfmt_destroy(/*@only@*/ yasm_dbgfmt *dbgfmt);

/** Generate debugging information bytecodes.
 * \param object        object
 * \param linemap       virtual/physical line mapping
 * \param errwarns      error/warning set
 * \note Errors and warnings are stored into errwarns.
 */
void yasm_dbgfmt_generate(yasm_object *object, yasm_linemap *linemap,
                          yasm_errwarns *errwarns);

#ifndef YASM_DOXYGEN

/* Inline macro implementations for dbgfmt functions */

#define yasm_dbgfmt_keyword(dbgfmt) \
    (((yasm_dbgfmt_base *)dbgfmt)->module->keyword)

#define yasm_dbgfmt_create(module, object) \
    module->create(object)

#define yasm_dbgfmt_destroy(dbgfmt) \
    ((yasm_dbgfmt_base *)dbgfmt)->module->destroy(dbgfmt)
#define yasm_dbgfmt_generate(object, linemap, ews) \
    ((yasm_dbgfmt_base *)((object)->dbgfmt))->module->generate \
        (object, linemap, ews)

#endif

#endif
