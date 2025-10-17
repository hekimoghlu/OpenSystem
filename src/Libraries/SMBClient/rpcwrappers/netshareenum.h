/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#ifndef _NETSHAREENUM_H_
#define _NETSHAREENUM_H_

#include <CoreFoundation/CoreFoundation.h>


#ifdef __cplusplus
extern "C" {
#endif
	
#define kNetCommentStrKey		CFSTR("NetCommentStr")
#define kNetShareTypeStrKey		CFSTR("NetShareTypeStr")
/*
 * share types
 */
#define	SMB_ST_DISK		0x0	/* A: */
#define	SMB_ST_PRINTER	0x1	/* LPT: */
#define	SMB_ST_COMM		0x2	/* COMM */
#define	SMB_ST_PIPE		0x3	/* IPC */
#define	SMB_ST_ANY		0x4	/* ????? */

	
int smb_netshareenum(SMBHANDLE inConnection, CFDictionaryRef *outDict, int DiskAndPrintSharesOnly);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _NETSHAREENUM_H_
