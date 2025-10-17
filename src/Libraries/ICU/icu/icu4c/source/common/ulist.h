/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
******************************************************************************
*   Copyright (C) 2009-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
******************************************************************************
*/

#ifndef ULIST_H
#define ULIST_H

#include "unicode/utypes.h"
#include "unicode/uenum.h"

struct UList;
typedef struct UList UList;

U_CAPI UList * U_EXPORT2 ulist_createEmptyList(UErrorCode *status);

U_CAPI void U_EXPORT2 ulist_addItemEndList(UList *list, const void *data, UBool forceDelete, UErrorCode *status);

U_CAPI void U_EXPORT2 ulist_addItemBeginList(UList *list, const void *data, UBool forceDelete, UErrorCode *status);

U_CAPI UBool U_EXPORT2 ulist_containsString(const UList *list, const char *data, int32_t length);

U_CAPI UBool U_EXPORT2 ulist_removeString(UList *list, const char *data);

U_CAPI void *U_EXPORT2 ulist_getNext(UList *list);

U_CAPI int32_t U_EXPORT2 ulist_getListSize(const UList *list);

U_CAPI void U_EXPORT2 ulist_resetList(UList *list);

U_CAPI void U_EXPORT2 ulist_deleteList(UList *list);

/*
 * The following are for use when creating UEnumeration object backed by UList.
 */
U_CAPI void U_EXPORT2 ulist_close_keyword_values_iterator(UEnumeration *en);

U_CAPI int32_t U_EXPORT2 ulist_count_keyword_values(UEnumeration *en, UErrorCode *status);

U_CAPI const char * U_EXPORT2 ulist_next_keyword_value(UEnumeration* en, int32_t *resultLength, UErrorCode* status);

U_CAPI void U_EXPORT2 ulist_reset_keyword_values_iterator(UEnumeration* en, UErrorCode* status);

U_CAPI UList * U_EXPORT2 ulist_getListFromEnum(UEnumeration *en);

#endif
