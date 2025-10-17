/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#ifndef _G_NACOMMON_H
#define _G_NACOMMON_H 1
/* .................................................. */

#include "tcl.h"
#include <ds.h>

/* .................................................. */

typedef enum attr_mode {
    A_LIST, A_GLOB, A_REGEXP, A_NONE
} attr_mode;

/* .................................................. */

void        gc_add    (GC* c, GCC* gx);
void        gc_remove (GC* c, GCC* gx);
void        gc_setup  (GC* c, GCC* gx, const char* name, G* g);
void        gc_delete (GC* c);
void        gc_rename (GC* c, GCC* gx, Tcl_Obj* newname, Tcl_Interp* interp);

int         gc_filter (int nodes, Tcl_Interp* interp,
		       int oc, Tcl_Obj* const* ov,
		       GCC* gx, GN_GET_GC* gf, G* g);

/* .................................................. */
#endif /* _G_NACOMMON_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
