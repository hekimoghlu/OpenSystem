/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
#ifndef _TCLREGEXP
#define _TCLREGEXP

#include "regex.h"

/*
 * The TclRegexp structure encapsulates a compiled regex_t, the flags that
 * were used to compile it, and an array of pointers that are used to indicate
 * subexpressions after a call to Tcl_RegExpExec. Note that the string and
 * objPtr are mutually exclusive. These values are needed by Tcl_RegExpRange
 * in order to return pointers into the original string.
 */

typedef struct TclRegexp {
    int flags;			/* Regexp compile flags. */
    regex_t re;			/* Compiled re, includes number of
				 * subexpressions. */
    CONST char *string;		/* Last string passed to Tcl_RegExpExec. */
    Tcl_Obj *objPtr;		/* Last object passed to Tcl_RegExpExecObj. */
    Tcl_Obj *globObjPtr;	/* Glob pattern rep of RE or NULL if none. */
    regmatch_t *matches;	/* Array of indices into the Tcl_UniChar
				 * representation of the last string matched
				 * with this regexp to indicate the location
				 * of subexpressions. */
    rm_detail_t details;	/* Detailed information on match (currently
				 * used only for REG_EXPECT). */
    int refCount;		/* Count of number of references to this
				 * compiled regexp. */
} TclRegexp;

#endif /* _TCLREGEXP */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
