/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Bell Laboratories
 *
 * fmtmode() and strperm() readonly data
 * for external format modes
 */

#include "modelib.h"

struct modeop	modetab[MODELEN] =
{
	0170000, 12, 0000000, 0, "-pc?d?b?-Cl?sDw?",
	0000400,  8, 0000000, 0, "-r",
	0000200,  7, 0000000, 0, "-w",
	0004000, 10, 0000100, 6, "-xSs",
	0000040,  5, 0000000, 0, "-r",
	0000020,  4, 0000000, 0, "-w",
#ifdef S_ICCTYP
	0003000,  8, 0000010, 3, "-x-xSs-x",
#else
	0002000,  9, 0000010, 3, "-xls",
#endif
	0000004,  2, 0000000, 0, "-r",
	0000002,  1, 0000000, 0, "-w",
#ifdef S_ICCTYP
	0003000,  8, 0000001, 0, "-xyY-xeE",
#else
	0001000,  8, 0000001, 0, "-xTt",
#endif
};

int	permmap[PERMLEN] =
{
	S_ISUID, X_ISUID,
	S_ISGID, X_ISGID,
	S_ISVTX, X_ISVTX,
	S_IRUSR, X_IRUSR,
	S_IWUSR, X_IWUSR,
	S_IXUSR, X_IXUSR,
	S_IRGRP, X_IRGRP,
	S_IWGRP, X_IWGRP,
	S_IXGRP, X_IXGRP,
	S_IROTH, X_IROTH,
	S_IWOTH, X_IWOTH,
	S_IXOTH, X_IXOTH
};
