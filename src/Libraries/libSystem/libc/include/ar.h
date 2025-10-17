/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#ifndef _AR_H_
#define	_AR_H_

/* Pre-4BSD archives had these magic numbers in them. */
#define	OARMAG1	0177555
#define	OARMAG2	0177545

#define	ARMAG		"!<arch>\n"	/* ar "magic number" */
#define	SARMAG		8		/* strlen(ARMAG); */

#define	AR_EFMT1	"#1/"		/* extended format #1 */

struct ar_hdr {
	char ar_name[16];		/* name */
	char ar_date[12];		/* modification time */
	char ar_uid[6];			/* user id */
	char ar_gid[6];			/* group id */
	char ar_mode[8];		/* octal file permissions */
	char ar_size[10];		/* size in bytes */
#define	ARFMAG	"`\n"
	char ar_fmag[2];		/* consistency check */
};

#endif /* !_AR_H_ */
