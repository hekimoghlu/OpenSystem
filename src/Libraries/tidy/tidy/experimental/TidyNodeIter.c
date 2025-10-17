/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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

#include "platform.h"
#include "tidy-int.h"

#include "TidyNodeIter.h"

TidyNodeIter *newTidyNodeIter( Node *pStart )
{
    TidyNodeIter *pThis = NULL;
    if (NULL != (pThis = MemAlloc( sizeof( TidyNodeIter ))))
    {
        ClearMemory( pThis, sizeof( TidyNodeIter ));
        pThis->pTop = pStart;
    }
    return pThis;
}

Node *nextTidyNode( TidyNodeIter *pThis )
{
    if (NULL == pThis->pCurrent)
    {
        // just starting out, initialize
        pThis->pCurrent = pThis->pTop->content;
    }
    else if (NULL != pThis->pCurrent->content)
    {
        // the next element, if any, is my first-born child
        pThis->pCurrent = pThis->pCurrent->content;
    }
    else 
    {
        // no children, I guess my next younger brother inherits the throne.
        while (   NULL == pThis->pCurrent->next
               && pThis->pTop != pThis->pCurrent->parent )
        {
            //  no siblings, do any of my ancestors have younger sibs?
            pThis->pCurrent = pThis->pCurrent->parent;
        }
        pThis->pCurrent = pThis->pCurrent->next;
    }
    return pThis->pCurrent;
}

void setCurrentNode( TidyNodeIter *pThis, Node *newCurr )
{
    if (NULL != newCurr)
        pThis->pCurrent = newCurr;
}
