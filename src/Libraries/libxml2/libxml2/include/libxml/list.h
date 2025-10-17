/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#ifndef __XML_LINK_INCLUDE__
#define __XML_LINK_INCLUDE__

#include <libxml/xmlversion.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _xmlLink xmlLink;
typedef xmlLink *xmlLinkPtr;

typedef struct _xmlList xmlList;
typedef xmlList *xmlListPtr;

/**
 * xmlListDeallocator:
 * @lk:  the data to deallocate
 *
 * Callback function used to free data from a list.
 */
typedef void (*xmlListDeallocator) (xmlLinkPtr lk);
/**
 * xmlListDataCompare:
 * @data0: the first data
 * @data1: the second data
 *
 * Callback function used to compare 2 data.
 *
 * Returns 0 is equality, -1 or 1 otherwise depending on the ordering.
 */
typedef int  (*xmlListDataCompare) (const void *data0, const void *data1);
/**
 * xmlListWalker:
 * @data: the data found in the list
 * @user: extra user provided data to the walker
 *
 * Callback function used when walking a list with xmlListWalk().
 *
 * Returns 0 to stop walking the list, 1 otherwise.
 */
typedef int (*xmlListWalker) (const void *data, void *user);

/* Creation/Deletion */
XMLPUBFUN xmlListPtr XMLCALL
		xmlListCreate		(xmlListDeallocator deallocator,
	                                 xmlListDataCompare compare);
XMLPUBFUN void XMLCALL
		xmlListDelete		(xmlListPtr l);

/* Basic Operators */
XMLPUBFUN void * XMLCALL
		xmlListSearch		(xmlListPtr l,
					 void *data);
XMLPUBFUN void * XMLCALL
		xmlListReverseSearch	(xmlListPtr l,
					 void *data);
XMLPUBFUN int XMLCALL
		xmlListInsert		(xmlListPtr l,
					 void *data) ;
XMLPUBFUN int XMLCALL
		xmlListAppend		(xmlListPtr l,
					 void *data) ;
XMLPUBFUN int XMLCALL
		xmlListRemoveFirst	(xmlListPtr l,
					 void *data);
XMLPUBFUN int XMLCALL
		xmlListRemoveLast	(xmlListPtr l,
					 void *data);
XMLPUBFUN int XMLCALL
		xmlListRemoveAll	(xmlListPtr l,
					 void *data);
XMLPUBFUN void XMLCALL
		xmlListClear		(xmlListPtr l);
XMLPUBFUN int XMLCALL
		xmlListEmpty		(xmlListPtr l);
XMLPUBFUN xmlLinkPtr XMLCALL
		xmlListFront		(xmlListPtr l);
XMLPUBFUN xmlLinkPtr XMLCALL
		xmlListEnd		(xmlListPtr l);
XMLPUBFUN int XMLCALL
		xmlListSize		(xmlListPtr l);

XMLPUBFUN void XMLCALL
		xmlListPopFront		(xmlListPtr l);
XMLPUBFUN void XMLCALL
		xmlListPopBack		(xmlListPtr l);
XMLPUBFUN int XMLCALL
		xmlListPushFront	(xmlListPtr l,
					 void *data);
XMLPUBFUN int XMLCALL
		xmlListPushBack		(xmlListPtr l,
					 void *data);

/* Advanced Operators */
XMLPUBFUN void XMLCALL
		xmlListReverse		(xmlListPtr l);
XMLPUBFUN void XMLCALL
		xmlListSort		(xmlListPtr l);
XMLPUBFUN void XMLCALL
		xmlListWalk		(xmlListPtr l,
					 xmlListWalker walker,
					 void *user);
XMLPUBFUN void XMLCALL
		xmlListReverseWalk	(xmlListPtr l,
					 xmlListWalker walker,
					 void *user);
XMLPUBFUN void XMLCALL
		xmlListMerge		(xmlListPtr l1,
					 xmlListPtr l2);
XMLPUBFUN xmlListPtr XMLCALL
		xmlListDup		(const xmlListPtr old);
XMLPUBFUN int XMLCALL
		xmlListCopy		(xmlListPtr cur,
					 const xmlListPtr old);
/* Link operators */
XMLPUBFUN void * XMLCALL
		xmlLinkGetData          (xmlLinkPtr lk);

/* xmlListUnique() */
/* xmlListSwap */

#ifdef __cplusplus
}
#endif

#endif /* __XML_LINK_INCLUDE__ */
