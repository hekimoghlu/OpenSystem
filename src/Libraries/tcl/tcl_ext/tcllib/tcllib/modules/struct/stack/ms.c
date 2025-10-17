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
#include <ms.h>
#include <m.h>
#include <s.h>
#include <util.h>

/* .................................................. */
/*
 *---------------------------------------------------------------------------
 *
 * stms_objcmd --
 *
 *	Implementation of stack objects, the main dispatcher function.
 *
 * Results:
 *	A standard Tcl result code.
 *
 * Side effects:
 *	Per the called methods.
 *
 *---------------------------------------------------------------------------
 */

int
stms_objcmd (ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* CONST* objv)
{
    S*  s = (S*) cd;
    int m;

    static CONST char* methods [] = {
	"clear", "destroy", "get",    "getr", "peek", "peekr",
	"pop",   "push",    "rotate", "size", "trim", "trim*",
	NULL
    };
    enum methods {
	M_CLEAR, M_DESTROY, M_GET,    M_GETR, M_PEEK, M_PEEKR,
	M_POP,   M_PUSH,    M_ROTATE, M_SIZE, M_TRIM, M_TRIMV
    };

    if (objc < 2) {
	Tcl_WrongNumArgs (interp, objc, objv, "option ?arg arg ...?");
	return TCL_ERROR;
    } else if (Tcl_GetIndexFromObj (interp, objv [1], methods, "option",
				    0, &m) != TCL_OK) {
	return TCL_ERROR;
    }

    /* Dispatch to methods. They check the #args in detail before performing
     * the requested functionality
     */

    switch (m) {
    case M_CLEAR:	return stm_CLEAR   (s, interp, objc, objv);
    case M_DESTROY:	return stm_DESTROY (s, interp, objc, objv);
    case M_GET:		return stm_GET     (s, interp, objc, objv, 0   ); /* get   */
    case M_GETR:	return stm_GET     (s, interp, objc, objv, 1   ); /* getr  */
    case M_PEEK:	return stm_PEEK    (s, interp, objc, objv, 0, 0); /* peek  */
    case M_PEEKR:	return stm_PEEK    (s, interp, objc, objv, 0, 1); /* peekr */
    case M_POP:		return stm_PEEK    (s, interp, objc, objv, 1, 0); /* pop   */
    case M_PUSH:	return stm_PUSH    (s, interp, objc, objv);
    case M_ROTATE:	return stm_ROTATE  (s, interp, objc, objv);
    case M_SIZE:	return stm_SIZE    (s, interp, objc, objv);
    case M_TRIM:	return stm_TRIM    (s, interp, objc, objv, 1   ); /* trim  */
    case M_TRIMV:	return stm_TRIM    (s, interp, objc, objv, 0   ); /* trim* */
    }
    /* Not coming to this place */
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
