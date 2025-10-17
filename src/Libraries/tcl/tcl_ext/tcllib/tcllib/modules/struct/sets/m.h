/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#ifndef _M_H
#define _M_H 1

#include "tcl.h"

int sm_ADD	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_CONTAINS	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_DIFFERENCE  (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_EMPTY	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_EQUAL	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_EXCLUDE	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_INCLUDE	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_INTERSECT   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_INTERSECT3  (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_SIZE        (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_SUBSETOF	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_SUBTRACT	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_SYMDIFF	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);
int sm_UNION	   (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv);

#endif /* _M_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
