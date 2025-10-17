/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#ifndef _OPTIONS_H_
#define _OPTIONS_H_
/*
 * argv[0] names. Used for tar and cpio emulation
 */

#define NM_TAR  "tar"
#define NM_CPIO "cpio"
#define NM_PAX  "pax"

/*
 * Constants used to specify the legal sets of flags in pax. For each major
 * operation mode of pax, a set of illegal flags is defined. If any one of
 * those illegal flags are found set, we scream and exit
 */
#define NONE	"none"

/*
 * flags (one for each option).
 */
#define	AF	0x00000001
#define	BF	0x00000002
#define	CF	0x00000004
#define	DF	0x00000008
#define	FF	0x00000010
#define	IF	0x00000020
#define	KF	0x00000040
#define	LF	0x00000080
#define	NF	0x00000100
#define	OF	0x00000200
#define	PF	0x00000400
#define	RF	0x00000800
#define	SF	0x00001000
#define	TF	0x00002000
#define	UF	0x00004000
#define	VF	0x00008000
#define	WF	0x00010000
#define	XF	0x00020000
#define	CBF	0x00040000	/* nonstandard extension */
#define	CDF	0x00080000	/* nonstandard extension */
#define	CEF	0x00100000	/* nonstandard extension */
#define	CGF	0x00200000	/* nonstandard extension */
#define	CHF	0x00400000	/* nonstandard extension */
#define	CLF	0x00800000	/* nonstandard extension */
#define	CPF	0x01000000	/* nonstandard extension */
#define	CTF	0x02000000	/* nonstandard extension */
#define	CUF	0x04000000	/* nonstandard extension */
#define	CXF	0x08000000
#define	CYF	0x10000000	/* nonstandard extension */
#define	CZF	0x20000000	/* nonstandard extension */
#ifdef __APPLE__
#define	C0F	0x40000000	/* nonstandard extension */
#endif

/*
 * ascii string indexed by bit position above (alter the above and you must
 * alter this string) used to tell the user what flags caused us to complain
 */
#ifndef __APPLE__
#define FLGCH	"abcdfiklnoprstuvwxBDEGHLPTUXYZ"
#else
#define FLGCH	"abcdfijklnoprstuvwxBDEGHLPTUXYZ0"
#endif /* __APPLE__*/

/*
 * legal pax operation bit patterns
 */

#define ISLIST(x)	(((x) & (RF|WF)) == 0)
#define	ISEXTRACT(x)	(((x) & (RF|WF)) == RF)
#define ISARCHIVE(x)	(((x) & (AF|RF|WF)) == WF)
#define ISAPPND(x)	(((x) & (AF|RF|WF)) == (AF|WF))
#define	ISCOPY(x)	(((x) & (RF|WF)) == (RF|WF))
#define	ISWRITE(x)	(((x) & (RF|WF)) == WF)

/*
 * Illegal option flag subsets based on pax operation
 */

#define	BDEXTR	(AF|BF|LF|TF|WF|XF|CBF|CHF|CLF|CPF|CXF)
#define	BDARCH	(CF|KF|LF|NF|PF|RF|CDF|CEF|CYF|CZF)
#ifndef __APPLE__
#define	BDCOPY	(AF|BF|FF|OF|XF|CBF|CEF)
#define	BDLIST (AF|BF|IF|KF|LF|OF|PF|RF|TF|UF|WF|XF|CBF|CDF|CHF|CLF|CPF|CXF|CYF|CZF)
#else
#define	BDCOPY	(AF|BF|FF|CBF|CEF)
#define	BDLIST (AF|BF|IF|KF|LF|PF|RF|TF|UF|WF|XF|CBF|CDF|CHF|CLF|CPF|CXF|CYF|CZF)
#endif /* __APPLE__*/

#endif /* _OPTIONS_H_ */
