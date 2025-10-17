/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#ifndef _DS_H
#define _DS_H 1

#include "tcl.h"

/* Forward declarations of references to queues.
 */

typedef struct Q* QPtr;

/* Queue structure
 */

typedef struct Q {
    Tcl_Command cmd; /* Token of the object command for
		      * the queue */
    Tcl_Obj* unget;  /* List object unget elements */
    Tcl_Obj* queue;  /* List object holding the main queue */
    Tcl_Obj* append; /* List object holding new elements */
    int at;          /* Index of next element to return from the main queue */
} Q;

#endif /* _DS_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
