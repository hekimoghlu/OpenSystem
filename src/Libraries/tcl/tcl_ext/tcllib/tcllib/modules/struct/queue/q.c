/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include <q.h>
#include <util.h>

/* .................................................. */

Q*
qu_new (void)
{
    Q* q = ALLOC (Q);

    q->at     = 0;
    q->unget  = Tcl_NewListObj (0,NULL);
    q->queue  = Tcl_NewListObj (0,NULL);
    q->append = Tcl_NewListObj (0,NULL);

    Tcl_IncrRefCount (q->unget); 
    Tcl_IncrRefCount (q->queue); 
    Tcl_IncrRefCount (q->append);

    return q;
}

void
qu_delete (Q* q)
{
    /* Delete a queue in toto.
     */

    Tcl_DecrRefCount (q->unget);
    Tcl_DecrRefCount (q->queue);
    Tcl_DecrRefCount (q->append);
    ckfree ((char*) q);
}

/* .................................................. */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
