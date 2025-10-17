/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef _G_NODE_H
#define _G_NODE_H 1

#include "tcl.h"
#include <ds.h>

void gn_shimmer  (Tcl_Obj* o, GN* n);
GN*  gn_get_node (G* g, Tcl_Obj* node, Tcl_Interp* interp, Tcl_Obj* graph);

#define gn_shimmer_self(n) \
    gn_shimmer ((n)->base.name, (n))

GN*  gn_new    (G* g, const char* name);
GN*  gn_dup    (G* dst, GN* src);
void gn_delete (GN* n);

void gn_err_duplicate (Tcl_Interp* interp, Tcl_Obj* n, Tcl_Obj* g);
void gn_err_missing   (Tcl_Interp* interp, Tcl_Obj* n, Tcl_Obj* g);

Tcl_Obj* gn_serial_arcs (GN* n, Tcl_Obj* empty, Tcl_HashTable* cn);

#endif /* _G_NODE_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
