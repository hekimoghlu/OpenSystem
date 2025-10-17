/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
 * $Id: edit_field.h,v 1.9 2013/06/08 15:46:01 tom Exp $
 *
 * Interface of edit_field.c
 */

#ifndef EDIT_FORM_H_incl
#define EDIT_FORM_H_incl 1

#include <form.h>

#define EDIT_FIELD(c) (MAX_FORM_COMMAND + c)

#define MY_HELP		EDIT_FIELD('h')
#define MY_QUIT		EDIT_FIELD('q')
#define MY_EDT_MODE	EDIT_FIELD('e')
#define MY_INS_MODE	EDIT_FIELD('t')

typedef struct {
    chtype background;
    int row_count;
    int *row_lengths;
} FieldAttrs;

extern FieldAttrs *field_attrs(FIELD * field);
extern void init_edit_field(FIELD * field, char *value);
extern void help_edit_field(void);
extern int edit_field(FORM * form, int *result);

#endif /* EDIT_FORM_H_incl */
