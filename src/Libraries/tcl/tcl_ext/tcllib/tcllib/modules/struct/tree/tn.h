/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#ifndef _TN_H
#define _TN_H 1

#include "tcl.h"
#include <ds.h>

void  tn_shimmer  (Tcl_Obj* o, TNPtr n);
TNPtr tn_get_node (TPtr t, Tcl_Obj* node, Tcl_Interp* interp, Tcl_Obj* tree);

TNPtr  tn_new	 (TPtr td, CONST char* name);
TNPtr  tn_dup	 (TPtr dst, TNPtr src);
void   tn_delete (TNPtr n);

void tn_node	  (TNPtr n);
void tn_notnode	  (TNPtr n);
void tn_leaf	  (TNPtr n);
void tn_notleaf	  (TNPtr n);
void tn_structure (TNPtr n, int depth);

void   tn_detach	 (TNPtr n);
TNPtr* tn_detachmany	 (TNPtr n, int len);
TNPtr* tn_detachchildren (TNPtr n, int* nc);

void tn_append	   (TNPtr p,	     TNPtr n);
void tn_insert	   (TNPtr p, int at, TNPtr n);

void tn_appendmany (TNPtr p,	     int nc, TNPtr* nv);
void tn_insertmany (TNPtr p, int at, int nc, TNPtr* nv);

void tn_cut	   (TNPtr n);

int	  tn_depth	    (TNPtr n);
int	  tn_height	    (TNPtr n);
int	  tn_ndescendants   (TNPtr n);
Tcl_Obj** tn_getdescendants (TNPtr n, int* nc);
Tcl_Obj** tn_getchildren    (TNPtr n, int* nc);

int tn_filternodes (int* nc,   Tcl_Obj** nv,
		    int	 cmdc, Tcl_Obj** cmdv,
		    Tcl_Obj* tree, Tcl_Interp* interp);

int tn_isancestorof (TNPtr a, TNPtr b);

void	 tn_extend_attr (TNPtr n);
void	 tn_set_attr	(TNPtr n, Tcl_Interp* interp, Tcl_Obj* dict);
Tcl_Obj* tn_get_attr	(TNPtr n, Tcl_Obj* empty);

int tn_serialize (TNPtr n, int listc, Tcl_Obj** listv,
		  int at, int parent, Tcl_Obj* empty);

#endif /* _TN_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
