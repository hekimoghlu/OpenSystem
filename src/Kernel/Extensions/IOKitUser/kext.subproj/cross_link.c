/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#include "cross_link.h"


/*********************************************************************
* Module Internal Variables
*********************************************************************/

static boolean_t __sCrossLinkEnabled  = FALSE;
static vm_size_t __sCrossLinkPageSize = 0;


/*********************************************************************
*********************************************************************/
boolean_t isCrossLinking(void)
{
    return __sCrossLinkEnabled;
}

/*********************************************************************
*********************************************************************/
boolean_t setCrossLinkPageSize(vm_size_t crossLinkPageSize)
{
    // verify radix 2
    if ((crossLinkPageSize != 0) && 
        ((crossLinkPageSize & (crossLinkPageSize - 1)) == 0)) {

        __sCrossLinkPageSize = crossLinkPageSize;
        __sCrossLinkEnabled = TRUE;

        return TRUE;   
    } else {
        return FALSE;
    }
}

/*********************************************************************
*********************************************************************/
vm_size_t getEffectivePageSize(void)
{
    if (__sCrossLinkEnabled) {
        return __sCrossLinkPageSize;
    } else {
        return PAGE_SIZE;
    }
}

/*********************************************************************
*********************************************************************/
vm_offset_t roundPageCrossSafe(vm_offset_t offset)
{
    // __sCrossLinkPageSize is checked for power of 2 above
    if (__sCrossLinkEnabled) {
        return (offset + (__sCrossLinkPageSize - 1)) & 
               (~(__sCrossLinkPageSize - 1));
    } else {
        return round_page(offset);
    }
}

/*********************************************************************
*********************************************************************/
mach_vm_offset_t roundPageCrossSafeFixedWidth(mach_vm_offset_t offset)
{
    // __sCrossLinkPageSize is checked for power of 2 above
    if (__sCrossLinkEnabled) {
        return (offset + (__sCrossLinkPageSize - 1)) & 
               (~(__sCrossLinkPageSize - 1));
    } else {
        return mach_vm_round_page(offset);
    }
}

