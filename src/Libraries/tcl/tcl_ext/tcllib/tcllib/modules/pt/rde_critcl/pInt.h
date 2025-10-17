/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#ifndef _P_INT_H
#define _P_INT_H 1

#include <p.h>     /* Public decls */
#include <param.h> /* PARAM architectural state */
#include <util.h>  /* Tracing support */

typedef struct RDE_STRING {
    struct RDE_STRING* next;
    Tcl_Obj*           self;
    int                id;
} RDE_STRING;

typedef struct RDE_STATE_ {
    RDE_PARAM   p;
    Tcl_Command c;

    struct RDE_STRING* sfirst;

    Tcl_HashTable str; /* Table to intern strings, i.e. convert them into
			* unique numerical indices for the PARAM instructions.
			*/

    /* And the counter mapping from ids to strings, this is handed to the
     * PARAM for use.
     */
    int    maxnum; /* NOTE -- */
    int    numstr; /* This is, essentially, an RDE_STACK (char* elements) */
    char** string; /* Convert over to that instead of replicating the code */

#ifdef RDE_TRACE
    int icount;  /* Instruction counter, when tracing */
#endif
} RDE_STATE_;

int param_intern (RDE_STATE p, char* literal);

#endif /* _P_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
