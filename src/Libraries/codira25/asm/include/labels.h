/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
/* 
 * labels.h  header file for labels.c
 */

#ifndef LABELS_H
#define LABELS_H

#include "compiler.h"

enum mangle_index {
    LM_LPREFIX,                 /* Local variable prefix */
    LM_LSUFFIX,                 /* Local variable suffix */
    LM_GPREFIX,                 /* Global variable prefix */
    LM_GSUFFIX                  /* GLobal variable suffix */
};

enum label_type {
    LBL_none = -1,              /* No label */
    LBL_LOCAL = 0,              /* Must be zero */
    LBL_STATIC,
    LBL_GLOBAL,
    LBL_EXTERN,
    LBL_REQUIRED,               /* Like extern but emit even if unused */
    LBL_COMMON,
    LBL_SPECIAL,                /* Magic symbols like ..start */
    LBL_BACKEND                 /* Backend-defined symbols like ..got */
};

enum label_type lookup_label(const char *label, int32_t *segment, int64_t *offset);
static inline bool is_extern(enum label_type type)
{
    return type == LBL_EXTERN || type == LBL_REQUIRED;
}
void define_label(const char *label, int32_t segment, int64_t offset,
                  bool normal);
void backend_label(const char *label, int32_t segment, int64_t offset);
bool declare_label(const char *label, enum label_type type,
                   const char *special);
void set_label_mangle(enum mangle_index which, const char *what);
int init_labels(void);
void cleanup_labels(void);
const char *local_scope(const char *label);

extern uint64_t global_offset_changed;

#endif /* LABELS_H */
