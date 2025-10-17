/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#ifndef __GenLinkedList__
#define __GenLinkedList__


#include <stddef.h>


struct  GenLinkedList
{
    void        *Head,
    *Tail;
    size_t LinkOffset;
};
typedef struct GenLinkedList GenLinkedList;


void        InitLinkedList( GenLinkedList *pList, size_t linkOffset);

void        AddToHead( GenLinkedList *pList, void *elem);
void        AddToTail( GenLinkedList *pList, void *elem);

int     RemoveFromList( GenLinkedList *pList, void *elem);

int     ReplaceElem( GenLinkedList *pList, void *elemInList, void *newElem);



struct  GenDoubleLinkedList
{
    void        *Head,
    *Tail;
    size_t FwdLinkOffset,
           BackLinkOffset;
};
typedef struct GenDoubleLinkedList GenDoubleLinkedList;


void        InitDoubleLinkedList( GenDoubleLinkedList *pList, size_t fwdLinkOffset,
                                  size_t backLinkOffset);

void        DLLAddToHead( GenDoubleLinkedList *pList, void *elem);

void        DLLRemoveFromList( GenDoubleLinkedList *pList, void *elem);



/* A GenLinkedOffsetList is like a GenLinkedList that stores the *Next field as a signed */
/* offset from the address of the beginning of the element, rather than as a pointer. */

struct  GenLinkedOffsetList
{
    size_t Head,
           Tail;
    size_t LinkOffset;
};
typedef struct GenLinkedOffsetList GenLinkedOffsetList;


void        InitLinkedOffsetList( GenLinkedOffsetList *pList, size_t linkOffset);

void        *GetHeadPtr( GenLinkedOffsetList *pList);
void        *GetTailPtr( GenLinkedOffsetList *pList);
void        *GetOffsetLink( GenLinkedOffsetList *pList, void *elem);

void        OffsetAddToHead( GenLinkedOffsetList *pList, void *elem);
void        OffsetAddToTail( GenLinkedOffsetList *pList, void *elem);

int     OffsetRemoveFromList( GenLinkedOffsetList *pList, void *elem);

int     OffsetReplaceElem( GenLinkedOffsetList *pList, void *elemInList, void *newElem);


#endif //	__GenLinkedList__
