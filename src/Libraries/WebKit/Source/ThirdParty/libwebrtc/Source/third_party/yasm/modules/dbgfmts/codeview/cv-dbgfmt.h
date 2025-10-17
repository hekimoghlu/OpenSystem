/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#ifndef YASM_CV_DBGFMT_H
#define YASM_CV_DBGFMT_H

typedef struct {
    char *pathname;             /* full pathname (drive+basepath+filename) */
    char *filename;             /* filename as yasm knows it internally */
    unsigned long str_off;      /* offset into pathname string table */
    unsigned long info_off;     /* offset into source info table */
    unsigned char digest[16];   /* MD5 digest of source file */
} cv_filename;

/* Global data */
typedef struct yasm_dbgfmt_cv {
    yasm_dbgfmt_base dbgfmt;        /* base structure */

    cv_filename *filenames;
    size_t filenames_size;
    size_t filenames_allocated;

    int version;
} yasm_dbgfmt_cv;

yasm_bytecode *yasm_cv__append_bc(yasm_section *sect, yasm_bytecode *bc);

/* Symbol/Line number functions */
yasm_section *yasm_cv__generate_symline
    (yasm_object *object, yasm_linemap *linemap, yasm_errwarns *errwarns);

/* Type functions */
yasm_section *yasm_cv__generate_type(yasm_object *object);

#endif
