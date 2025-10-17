/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
 * Getty description definitions.
 */
struct	gettystrs {
	const char *field;	/* name to lookup in gettytab */
	char	*defalt;	/* value we find by looking in defaults */
	char	*value;		/* value that we find there */
};

struct	gettynums {
	const char *field;	/* name to lookup */
	long	defalt;		/* number we find in defaults */
	long	value;		/* number we find there */
	int	set;		/* we actually got this one */
};

struct gettyflags {
	const char *field;	/* name to lookup */
	char	invrt;		/* name existing in gettytab --> false */
	char	defalt;		/* true/false in defaults */
	char	value;		/* true/false flag */
	char	set;		/* we found it */
};

/*
 * String values.
 */
#define	NX	gettystrs[0].value
#define	CL	gettystrs[1].value
#define IM	gettystrs[2].value
#define	LM	gettystrs[3].value
#define	ER	gettystrs[4].value
#define	KL	gettystrs[5].value
#define	ET	gettystrs[6].value
#define	PC	gettystrs[7].value
#define	TT	gettystrs[8].value
#define	EV	gettystrs[9].value
#define	LO	gettystrs[10].value
#define HN	gettystrs[11].value
#define HE	gettystrs[12].value
#define IN	gettystrs[13].value
#define QU	gettystrs[14].value
#define XN	gettystrs[15].value
#define XF	gettystrs[16].value
#define BK	gettystrs[17].value
#define SU	gettystrs[18].value
#define DS	gettystrs[19].value
#define RP	gettystrs[20].value
#define FL	gettystrs[21].value
#define WE	gettystrs[22].value
#define LN	gettystrs[23].value
#define Lo	gettystrs[24].value
#define PP	gettystrs[25].value
#define IF	gettystrs[26].value
#define IC	gettystrs[27].value
#define AC	gettystrs[28].value
#define AL	gettystrs[29].value
#define DF	gettystrs[30].value
#define IMP	gettystrs[31].value

/*
 * Numeric definitions.
 */
#define	IS	gettynums[0].value
#define	OS	gettynums[1].value
#define	SP	gettynums[2].value
#define	ND	gettynums[3].value
#define	CD	gettynums[4].value
#define	TD	gettynums[5].value
#define	FD	gettynums[6].value
#define	BD	gettynums[7].value
#define	TO	gettynums[8].value
#define	F0	gettynums[9].value
#define	F0set	gettynums[9].set
#define	F1	gettynums[10].value
#define	F1set	gettynums[10].set
#define	F2	gettynums[11].value
#define	F2set	gettynums[11].set
#define	PF	gettynums[12].value
#define	C0	gettynums[13].value
#define	C0set	gettynums[13].set
#define	C1	gettynums[14].value
#define	C1set	gettynums[14].set
#define	C2	gettynums[15].value
#define	C2set	gettynums[15].set
#define	I0	gettynums[16].value
#define	I0set	gettynums[16].set
#define	I1	gettynums[17].value
#define	I1set	gettynums[17].set
#define	I2	gettynums[18].value
#define	I2set	gettynums[18].set
#define	L0	gettynums[19].value
#define	L0set	gettynums[19].set
#define	L1	gettynums[20].value
#define	L1set	gettynums[20].set
#define	L2	gettynums[21].value
#define	L2set	gettynums[21].set
#define	O0	gettynums[22].value
#define	O0set	gettynums[22].set
#define	O1	gettynums[23].value
#define	O1set	gettynums[23].set
#define	O2	gettynums[24].value
#define	O2set	gettynums[24].set
#define	DE	gettynums[25].value
#define RTset	gettynums[26].set
#define RT	gettynums[26].value
#define CT	gettynums[27].value
#define DC	gettynums[28].value

/*
 * Boolean values.
 */
#define	HT	gettyflags[0].value
#define	NL	gettyflags[1].value
#define	EP	gettyflags[2].value
#define	EPset	gettyflags[2].set
#define	OP	gettyflags[3].value
#define	OPset	gettyflags[3].set
#define	AP	gettyflags[4].value
#define	APset	gettyflags[4].set
#define	EC	gettyflags[5].value
#define	CO	gettyflags[6].value
#define	CB	gettyflags[7].value
#define	CK	gettyflags[8].value
#define	CE	gettyflags[9].value
#define	PE	gettyflags[10].value
#define	RW	gettyflags[11].value
#define	XC	gettyflags[12].value
#define	LC	gettyflags[13].value
#define	UC	gettyflags[14].value
#define	IG	gettyflags[15].value
#define	PS	gettyflags[16].value
#define	HC	gettyflags[17].value
#define UB	gettyflags[18].value
#define AB	gettyflags[19].value
#define DX	gettyflags[20].value
#define	NP	gettyflags[21].value
#define	NPset	gettyflags[21].set
#define	MB	gettyflags[22].value
#define	HW	gettyflags[23].value
#define	NC	gettyflags[24].value
#define	PL	gettyflags[25].value
