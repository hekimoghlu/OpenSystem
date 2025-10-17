/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
 *	nfs_subr.h
 *
 *	Copyright (c) 1996 Sun Microsystems Inc
 *	All Rights Reserved.
 */

#ifndef _NFS_SUBR_H
#define _NFS_SUBR_H

#pragma ident	"@(#)nfs_subr.h	1.3	05/06/08 SMI"

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * nfs library routines
 */
extern void URLparse(char *);
extern int convert_special(char **, char *, char *, char *, char *);

#ifdef __cplusplus
}
#endif

#endif  /* _NFS_SUBR_H */
