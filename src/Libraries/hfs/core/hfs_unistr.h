/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#ifndef __HFS_UNISTR__
#define __HFS_UNISTR__

#include <sys/types.h>

/* 
 * hfs_unitstr.h
 *
 * This file contains definition of the unicode string used for HFS Plus 
 * files and folder names, as described by the on-disk format.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif


#ifndef _HFSUNISTR255_DEFINED_
#define _HFSUNISTR255_DEFINED_
/* Unicode strings are used for HFS Plus file and folder names */
struct HFSUniStr255 {
	u_int16_t	length;		/* number of unicode characters */
	u_int16_t	unicode[255];	/* unicode characters */
} __attribute__((aligned(2), packed));
typedef struct HFSUniStr255 HFSUniStr255;
typedef const HFSUniStr255 *ConstHFSUniStr255Param;
#endif /* _HFSUNISTR255_DEFINED_ */


#ifdef __cplusplus
}
#endif


#endif /* __HFS_UNISTR__ */
