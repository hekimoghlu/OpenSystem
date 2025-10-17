/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#ifndef _SECITEM_H_
#define _SECITEM_H_
/*
 * secitem.h - public data structures and prototypes for handling
 *	       SECItems
 */

#include <security_asn1/plarenas.h>
#include <security_asn1/seccomon.h>
#include "plhash.h"

SEC_BEGIN_PROTOS

/*
** Allocate an item.  If "arena" is not NULL, then allocate from there,
** otherwise allocate from the heap.  If "item" is not NULL, allocate
** only the data for the item, not the item itself.  The item structure
** is allocated zero-filled; the data buffer is not zeroed.
**
** The resulting item is returned; NULL if any error occurs.
**
** XXX This probably should take a SECItemType, but since that is mostly
** unused and our improved APIs (aka Stan) are looming, I left it out.
*/
extern SECItem* SECITEM_AllocItem(PRArenaPool* arena, SECItem* item, size_t len);

/*
** Reallocate the data for the specified "item".  If "arena" is not NULL,
** then reallocate from there, otherwise reallocate from the heap.
** In the case where oldlen is 0, the data is allocated (not reallocated).
** In any case, "item" is expected to be a valid SECItem pointer;
** SECFailure is returned if it is not.  If the allocation succeeds,
** SECSuccess is returned.
*/
extern SECStatus
SECITEM_ReallocItem(PRArenaPool* arena, SECItem* item, unsigned int oldlen, unsigned int newlen);

/*
** Compare two items returning the difference between them.
*/
extern SECComparison SECITEM_CompareItem(const SECItem* a, const SECItem* b);

/*
** Compare two items -- if they are the same, return true; otherwise false.
*/
extern Boolean SECITEM_ItemsAreEqual(const SECItem* a, const SECItem* b);

/*
** Copy "from" to "to"
*/
extern SECStatus SECITEM_CopyItem(PRArenaPool* arena, SECItem* to, const SECItem* from);

/*
** Allocate an item and copy "from" into it.
*/
extern SECItem* SECITEM_DupItem(const SECItem* from);

/*
 ** Allocate an item and copy "from" into it.  The item itself and the 
 ** data it points to are both allocated from the arena.  If arena is
 ** NULL, this function is equivalent to SECITEM_DupItem.
 */
extern SECItem* SECITEM_ArenaDupItem(PRArenaPool* arena, const SECItem* from);

/*
** Free "zap". If freeit is PR_TRUE then "zap" itself is freed.
*/
extern void SECITEM_FreeItem(SECItem* zap, Boolean freeit);

/*
** Zero and then free "zap". If freeit is PR_TRUE then "zap" itself is freed.
*/
extern void SECITEM_ZfreeItem(SECItem* zap, Boolean freeit);

PLHashNumber PR_CALLBACK SECITEM_Hash(const void* key);

PRIntn PR_CALLBACK SECITEM_HashCompare(const void* k1, const void* k2);


SEC_END_PROTOS

#endif /* _SECITEM_H_ */
