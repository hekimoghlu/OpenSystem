/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
	File:		WindowsTypesForMac.h

	Contains:	Define common Windows data types in mac terms.

	Written by:	Doug Mitchell

	Copyright: (c) 2000 by Apple Computer, Inc., all rights reserved.

	Change History (most recent first):

		02/10/99	dpm		Created.
 
*/

#ifndef	_WINDOWS_TYPES_FOR_MAC_H_
#define _WINDOWS_TYPES_FOR_MAC_H_

#include <sys/types.h>

typedef u_int8_t 	UCHAR;
typedef int8_t 	CHAR;
typedef u_int8_t 	BYTE;
typedef char	TCHAR;
typedef int16_t	WORD;
typedef int32_t	DWORD;
typedef u_int16_t	USHORT;
typedef u_int32_t	ULONG;
typedef int32_t	LONG;
typedef u_int32_t	UINT;
typedef int64_t	LONGLONG;
typedef u_int8_t	*LPBYTE;
typedef int8_t 	*LPSTR;
typedef int16_t	*LPWORD;
typedef	int8_t	*LPCTSTR;		/* ??? */
typedef	int8_t	*LPCSTR;		/* ??? */
typedef void	*LPVOID;
typedef void	*HINSTANCE;
typedef	void	*HANDLE;

#define WINAPI

#endif	/* _WINDOWS_TYPES_FOR_MAC_H_*/

