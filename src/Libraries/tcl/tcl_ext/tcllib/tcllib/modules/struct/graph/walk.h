/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#ifndef _G_WALK_H
#define _G_WALK_H 1
/* .................................................. */

#include "tcl.h"
#include <ds.h>

#define W_USAGE "node ?-dir forward|backward? ?-order pre|post|both? ?-type bfs|dfs? -command cmd"

/* .................................................. */

enum wtypes {
    WG_BFS, WG_DFS
};

enum worder {
    WO_BOTH, WO_PRE, WO_POST
};

enum wdir {
    WD_BACKWARD, WD_FORWARD
};

int g_walkoptions (Tcl_Interp* interp,
		   int objc, Tcl_Obj* const* objv,
		   int* type, int* order, int* dir,
		   int* cc, Tcl_Obj*** cv);

int g_walk (Tcl_Interp* interp, Tcl_Obj* go, GN* n,
	    int type, int order, int dir,
	    int cc, Tcl_Obj** cv);

/* .................................................. */
#endif /* _G_WALK_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
