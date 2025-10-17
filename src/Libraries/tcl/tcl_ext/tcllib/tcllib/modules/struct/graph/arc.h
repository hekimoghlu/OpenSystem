/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
#ifndef _G_ARC_H
#define _G_ARC_H 1

#include "tcl.h"
#include <ds.h>

void ga_shimmer (Tcl_Obj* o, GA* a);
GA*  ga_get_arc (G* g, Tcl_Obj* arc, Tcl_Interp* interp, Tcl_Obj* graph);

#define ga_shimmer_self(a) \
    ga_shimmer ((a)->base.name, (a))

GA*  ga_new    (G* g, const char* name, GN* src, GN* dst);
GA*  ga_dup    (G* dst, GA* src);
void ga_delete (GA* a);

void ga_arc	  (GA* a);
void ga_notarc	  (GA* a);

void ga_mv_src (GA* a, GN* nsrc);
void ga_mv_dst (GA* a, GN* ndst);

void ga_err_duplicate (Tcl_Interp* interp, Tcl_Obj* a, Tcl_Obj* g);
void ga_err_missing   (Tcl_Interp* interp, Tcl_Obj* a, Tcl_Obj* g);

Tcl_Obj* ga_serial (GA* a, Tcl_Obj* empty, int nodeId);

#endif /* _G_ARC_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
