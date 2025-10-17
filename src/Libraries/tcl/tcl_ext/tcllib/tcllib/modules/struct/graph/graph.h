/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
#ifndef _G_GRAPH_H
#define _G_GRAPH_H 1
/* .................................................. */

#include "tcl.h"
#include <ds.h>

/* .................................................. */

G*          g_new         (void);
void        g_delete      (G* g);

const char* g_newnodename (G* g);
const char* g_newarcname  (G* g);

Tcl_Obj*    g_serialize   (Tcl_Interp* interp, Tcl_Obj* go,
			   G* g, int oc, Tcl_Obj* const* ov);
int         g_deserialize (G* dst, Tcl_Interp* interp, Tcl_Obj* src);
int         g_assign      (G* dst, G* src);

Tcl_Obj*    g_ms_serialize (Tcl_Interp* interp, Tcl_Obj* go, G* g,
			    int oc, Tcl_Obj* const* ov);
int	    g_ms_set       (Tcl_Interp* interp, Tcl_Obj* go, G* g,
			    Tcl_Obj* dst);
int	    g_ms_assign    (Tcl_Interp* interp, G* g, Tcl_Obj* src);

/* .................................................. */
#endif /* _G_GRAPH_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
