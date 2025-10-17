/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#ifndef YASM_PARSER_H
#define YASM_PARSER_H

/** YASM parser module interface.  The "front end" of the assembler. */
typedef struct yasm_parser_module {
    /** One-line description of the parser */
    const char *name;

    /** Keyword used to select parser on the command line */
    const char *keyword;

    /** NULL-terminated list of preprocessors that are valid to use with this
     * parser.  The raw preprocessor (raw_preproc) should always be in this
     * list so it's always possible to have no preprocessing done.
     */
    const char **preproc_keywords;

    /** Default preprocessor. */
    const char *default_preproc_keyword;

    /** NULL-terminated list of standard macro lookups.  NULL if none. */
    const yasm_stdmac *stdmacs;

    /** Parse a source file into an object.
     * \param object    object to parse into (already created)
     * \param pp        preprocessor
     * \param save_input        nonzero if the parser should save the original
     *                          lines of source into the object's linemap (via
     *                          yasm_linemap_add_data()).
     * \param errwarns  error/warning set
     * \note Parse errors and warnings are stored into errwarns.
     */
    void (*do_parse)
        (yasm_object *object, yasm_preproc *pp, int save_input,
         yasm_linemap *linemap, yasm_errwarns *errwarns);
} yasm_parser_module;

#endif
