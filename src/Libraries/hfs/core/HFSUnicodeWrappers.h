/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
	File:		HFSUnicodeWrappers.h

	Contains:	IPI to Unicode routines used by File Manager.

	Version:	HFS Plus 1.0

	Written by:	Mark Day

	Copyright:	(c) 1996-1997 by Apple Inc., all rights reserved.

	File Ownership:

		DRI:				xxx put dri here xxx

		Other Contact:		xxx put other contact here xxx

		Technology:			xxx put technology here xxx

	Writers:

		(DSH)	Deric Horn
		(msd)	Mark Day
		(djb)	Don Brady

	Change History (most recent first):

	  <CS11>	11/16/97	djb		Change Unicode.h to UnicodeConverter.h.
	  <CS10>	 11/7/97	msd		Remove prototype for CompareUnicodeNames(). Add prototype for
									FastUnicodeCompare().
	   <CS9>	10/13/97	djb		Add encoding/index macros and add prototypes for new Get/Set
									encodding routines.
	   <CS8>	 9/15/97	djb		InitUnicodeConverter now takes a boolean.
	   <CS7>	 9/10/97	msd		Add prototype for InitializeEncodingContext.
	   <CS6>	 6/26/97	DSH		Include  "MockConverter" prototype for DFA usage.
	   <CS5>	 6/25/97	DSH		Removed Prototype definitions, and checked in Unicode.h and
									TextCommon.h from Julio Gonzales into InternalInterfaces.
	   <CS4>	 6/25/97	msd		Add prototypes for some new Unicode routines that haven't
									appeared in MasterInterfaces yet.
	   <CS3>	 6/18/97	djb		Add more ConversionContexts routines.
	   <CS2>	 6/13/97	djb		Switched to ConvertUnicodeToHFSName, ConvertHFSNameToUnicode, &
									CompareUnicodeNames.
	   <CS1>	 4/28/97	djb		first checked in
	  <HFS1>	12/12/96	msd		first checked in

*/
#ifndef _HFSUNICODEWRAPPERS_
#define _HFSUNICODEWRAPPERS_

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE

#include "hfs_macos_defs.h"
#include "hfs_format.h"


extern OSErr ConvertUnicodeToUTF8Mangled ( ByteCount srcLen,
									ConstUniCharArrayPtr srcStr,
									ByteCount maxDstLen,
					 				ByteCount *actualDstLen,
									unsigned char* dstStr ,
									HFSCatalogNodeID cnid);

/*
	This routine compares two Unicode names based on an ordering defined by the HFS Plus B-tree.
	This ordering must stay fixed for all time.
	
	Output:
		-n		name1 < name2	(i.e. name 1 sorts before name 2)
		 0		name1 = name2
		+n		name1 > name2
	
	NOTE: You should not depend on the magnitude of the result, just its sign.  That is, when name1 < name2, then any
	negative number may be returned.
*/

extern int32_t FastUnicodeCompare(register ConstUniCharArrayPtr str1, register ItemCount length1,
								 register ConstUniCharArrayPtr str2, register ItemCount length2);

extern int32_t UnicodeBinaryCompare (register ConstUniCharArrayPtr str1, register ItemCount length1,
								 register ConstUniCharArrayPtr str2, register ItemCount length2);

extern int32_t FastRelString( ConstStr255Param str1, ConstStr255Param str2 );


extern HFSCatalogNodeID GetEmbeddedFileID( ConstStr31Param filename, u_int32_t length, u_int32_t *prefixLength );
extern u_int32_t CountFilenameExtensionChars( const unsigned char * filename, u_int32_t length );

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* _HFSUNICODEWRAPPERS_ */
