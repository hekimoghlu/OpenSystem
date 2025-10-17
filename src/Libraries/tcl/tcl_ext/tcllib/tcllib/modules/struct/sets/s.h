/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#ifndef _S_H
#define _S_H 1

#include "tcl.h"
#include "ds.h"

int      s_get (Tcl_Interp* interp, Tcl_Obj* o, SPtr* sStar);
Tcl_Obj* s_new (SPtr s);

Tcl_ObjType* s_stype (void);
Tcl_ObjType* s_ltype (void);

void s_add        (SPtr a, SPtr b, int* newPtr);
void s_add1       (SPtr a, const char* item);
int  s_contains   (SPtr a, const char* item);
SPtr s_difference (SPtr a, SPtr b);
SPtr s_dup        (SPtr a); /* a == NULL allowed */
int  s_empty      (SPtr a);
int  s_equal      (SPtr a, SPtr b);
void s_free       (SPtr a);
SPtr s_intersect  (SPtr a, SPtr b);
int  s_size       (SPtr a);
int  s_subsetof   (SPtr a, SPtr b);
void s_subtract   (SPtr a, SPtr b, int* delPtr);
void s_subtract1  (SPtr a, const char* item);
SPtr s_union      (SPtr a, SPtr b);

#endif /* _S_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
