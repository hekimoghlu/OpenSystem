/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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
#ifndef _T_H
#define _T_H 1

#include "tcl.h"
#include <ds.h>

TPtr t_new	 (void);
void t_delete	 (TPtr t);
void t_structure (TPtr t);
void t_dump      (TPtr t, FILE* f);

int  t_deserialize (TPtr dst, Tcl_Interp* interp, Tcl_Obj* src);
int  t_assign	   (TPtr dst, TPtr src);

enum wtypes {
    WT_BFS, WT_DFS
};

enum worder {
    WO_BOTH, WO_IN, WO_PRE, WO_POST
};

typedef int (*t_walk_function) (Tcl_Interp* interp,
				TN* n, Tcl_Obj* cs,
				Tcl_Obj* da, Tcl_Obj* db,
				Tcl_Obj* action);

int t_walkoptions (Tcl_Interp* interp, int n,
		   int objc, Tcl_Obj* CONST* objv,
		   int* type, int* order, int* remainder,
		   char* usage);

int t_walk (Tcl_Interp* interp, TN* tdn, int type, int order,
	    t_walk_function f, Tcl_Obj* cs,
	    Tcl_Obj* avn, Tcl_Obj* nvn);

int t_walk_invokescript (Tcl_Interp* interp, TN* n, Tcl_Obj* cs,
			 Tcl_Obj* avn, Tcl_Obj* nvn,
			 Tcl_Obj* action);

int t_walk_invokecmd (Tcl_Interp* interp, TN* n, Tcl_Obj* dummy0,
		      Tcl_Obj* dummy1, Tcl_Obj* dummy2,
		      Tcl_Obj* action);

CONST char* t_newnodename (T* td);

#endif /* _T_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
