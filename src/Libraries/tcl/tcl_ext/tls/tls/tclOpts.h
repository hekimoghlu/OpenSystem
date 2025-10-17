/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#ifndef _TCL_OPTS_H
#define _TCL_OPTS_H

#define OPT_PROLOG(option)			\
    if (strcmp(opt, (option)) == 0) {		\
	if (++idx >= objc) {			\
	    Tcl_AppendResult(interp,		\
		"no argument given for ",	\
		(option), " option",		\
		(char *) NULL);			\
	    return TCL_ERROR;			\
	}
#define OPT_POSTLOG()				\
	continue;				\
    }
#define OPTOBJ(option, var)			\
    OPT_PROLOG(option)				\
    var = objv[idx];				\
    OPT_POSTLOG()

#define OPTSTR(option, var)			\
    OPT_PROLOG(option)				\
    var = Tcl_GetStringFromObj(objv[idx], NULL);\
    OPT_POSTLOG()

#define OPTINT(option, var)			\
    OPT_PROLOG(option)				\
    if (Tcl_GetIntFromObj(interp, objv[idx],	\
	    &(var)) != TCL_OK) {		\
	    return TCL_ERROR;			\
    }						\
    OPT_POSTLOG()

#define OPTBOOL(option, var)			\
    OPT_PROLOG(option)				\
    if (Tcl_GetBooleanFromObj(interp, objv[idx],\
	    &(var)) != TCL_OK) {		\
	    return TCL_ERROR;			\
    }						\
    OPT_POSTLOG()

#define OPTBAD(type, list)			\
    Tcl_AppendResult(interp, "bad ", (type),	\
		" \"", opt, "\": must be ",	\
		(list), (char *) NULL)

#endif /* _TCL_OPTS_H */
