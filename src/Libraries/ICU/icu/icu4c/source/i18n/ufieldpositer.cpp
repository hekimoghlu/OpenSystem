/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
*****************************************************************************************
* Copyright (C) 2015, International Business Machines
* Corporation and others. All Rights Reserved.
*****************************************************************************************
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/ufieldpositer.h"
#include "unicode/fpositer.h"
#include "unicode/localpointer.h"

U_NAMESPACE_USE


U_CAPI UFieldPositionIterator* U_EXPORT2
ufieldpositer_open(UErrorCode* status)
{
    if (U_FAILURE(*status)) {
        return nullptr;
    }
    FieldPositionIterator* fpositer = new FieldPositionIterator();
    if (fpositer == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
    }
    return (UFieldPositionIterator*)fpositer;
}


U_CAPI void U_EXPORT2
ufieldpositer_close(UFieldPositionIterator *fpositer)
{
    delete (FieldPositionIterator*)fpositer;
}


U_CAPI int32_t U_EXPORT2
ufieldpositer_next(UFieldPositionIterator *fpositer,
                   int32_t *beginIndex, int32_t *endIndex)
{
    FieldPosition fp;
    int32_t field = -1;
    if (((FieldPositionIterator*)fpositer)->next(fp)) {
        field = fp.getField();
        if (beginIndex) {
            *beginIndex = fp.getBeginIndex();
        }
        if (endIndex) {
            *endIndex = fp.getEndIndex();
        }
    }
    return field;
}


#endif /* #if !UCONFIG_NO_FORMATTING */
