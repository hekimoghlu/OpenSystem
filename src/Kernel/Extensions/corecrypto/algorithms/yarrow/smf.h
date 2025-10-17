/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
	File:		smf.h

	Contains:	Secure malloc/free API.

	Written by:	Doug Mitchell

	Copyright: (c) 2000 by Apple Computer, Inc., all rights reserved.

	Change History (most recent first):

		02/10/00	dpm		Created, based on Counterpane's Yarrow code. 
 
*/

#ifndef _YARROW_SMF_H_
#define _YARROW_SMF_H_

#if defined(__cplusplus)
extern "C" {
#endif

/* smf.h */

	/*  
	Header file for secure malloc and free routines used by the Counterpane
	PRNG. Use this code to set up a memory-mapped file out of the system 
	paging file, allocate and free memory from it, and then return
	the memory to the system registry after having securely overwritten it.
	Details of the secure overwrite can be found in Gutmann 1996 (Usenix).
	Trying to explain it here will cause my head to begin to hurt.
	Ari Benbasat (pigsfly@unixg.ubc.ca)
	*/



#if		defined(macintosh) || defined(__APPLE__)
#include "macOnly.h"
#define MMPTR	void *

#ifndef SMFAPI 
#define SMFAPI 
#endif

#else	/* original Yarrow */

/* Declare HOOKSAPI as __declspec(dllexport) before
   including this file in the actual DLL */
#ifndef SMFAPI 
#define SMFAPI __declspec(dllimport)
#endif
#define MMPTR	BYTE

#endif /* macintosh */


#define MM_NULL	((void *)0)

/* Function forward declarations */
SMFAPI void mmInit( void );
SMFAPI MMPTR mmMalloc(DWORD request);
SMFAPI void mmFree(MMPTR ptrnum);
SMFAPI LPVOID mmGetPtr(MMPTR ptrnum);
SMFAPI void mmReturnPtr(MMPTR ptrnum);
#if	0
SMFAPI void mmFreePtr(LPVOID ptr);
#endif

#if defined(__cplusplus)
}
#endif

#endif	/* _YARROW_SMF_H_*/
