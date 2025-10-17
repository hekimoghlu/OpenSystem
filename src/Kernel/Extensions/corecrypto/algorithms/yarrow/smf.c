/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
/*
	File:		smf.c

	Contains:	platform-dependent malloc/free
 
*/

#include "smf.h"
#include <sys/malloc.h>
#include <sys/systm.h>


SMFAPI void mmInit( void )
{
	return;
}

SMFAPI MMPTR mmMalloc(DWORD request)
{
    // since kfree requires that we pass in the alloc size, add enough bytes to store a dword
    void* mem;
    
    mem = _MALLOC (request, M_TEMP, M_WAITOK);
    
    if (mem == 0) // oops, it didn't appear to work
    {
        printf ("Couldn't allocate kernel memory!\n");
        return NULL;
    }
    
    return (MMPTR) mem;
}

SMFAPI void mmFree(MMPTR ptrnum)
{
    // get the size of the pointer back
    _FREE (ptrnum, M_TEMP);
}

SMFAPI LPVOID mmGetPtr(MMPTR ptrnum)
{
	return (LPVOID)ptrnum;
}

SMFAPI void mmReturnPtr(__unused MMPTR ptrnum)
{
	/* nothing */
	return;
}

