/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include "nssUtils.h"
#include <string.h>

/*
 * Compare two SecAsn1Items (or two SecAsn1Oids), return true if identical.
 */
int nssCompareSecAsn1Items(const SecAsn1Item* data1, const SecAsn1Item* data2)
{
    if ((data1 == NULL) || (data1->Data == NULL) || (data2 == NULL) ||
        (data2->Data == NULL) || (data1->Length != data2->Length)) {
        return 0;
    }
    if (data1->Length != data2->Length) {
        return 0;
    }
    return memcmp(data1->Data, data2->Data, data1->Length) == 0;
}

int nssCompareCssmData(const SecAsn1Item* data1, const SecAsn1Item* data2)
{
    return nssCompareSecAsn1Items(data1, data2);
}

/*
 * How many items in a NULL-terminated array of pointers?
 */
unsigned nssArraySize(const void** array)
{
    unsigned count = 0;
    if (array) {
        while (*array++) {
            count++;
        }
    }
    return count;
}
