/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#include "lexer.h"

typedef struct _TidyNodeIter
{
    Node *pTop, *pCurrent;
} TidyNodeIter;

TidyNodeIter *newTidyNodeIter( Node *pStart );

/* 
    nextTidyNode( TidyNodeIter *pIter )

    if pCurrent is NULL, this function initializes it to match pTop, and
    returns that value, otherwise it advances to the next node in order, 
    and returns that value. When pTop == pCurrent, the function returns NULL
    to indicate that the entire tree has been visited.
*/
Node *nextTidyNode( TidyNodeIter *pIter );

/*
    setCurrentNode( TidyNodeIter *pThis, Node *newCurr )

    Resets pCurrent to match the passed value; useful if you need to back up
    to an unaltered point in the tree, or to skip a section. The next call to 
    nextTidyNode() will return the node which follows newCurr in order.

    Minimal error checking is performed; unexpected results _will_ occur if 
    newCurr is not a descendant node of pTop.
*/
void setCurrentNode( TidyNodeIter *pThis, Node *newCurr );
