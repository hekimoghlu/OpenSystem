/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#ifndef _SHNODES_H
#define _SHNODES_H	1
/*
 *	UNIX shell
 *	Written by David Korn
 *
 */


#include	<ast.h>
#include	"argnod.h"

/* command tree for tretyp */
#define FINT		(02<<COMBITS)		/* non-interruptable */
#define FAMP		(04<<COMBITS)		/* background */
#define FPIN		(010<<COMBITS)		/* input is a pipe */
#define FPOU		(040<<COMBITS)		/* output is a pipe */
#define FPCL		(0100<<COMBITS)		/* close the pipe */
#define FCOOP		(0200<<COMBITS)		/* cooperating process */
#define FSHOWME		(0400<<COMBITS)		/* set for showme commands  */
#define FALTPIPE	(02000<<COMBITS)	/* alternate pipes &| */
#define FPOSIX		(02<<COMBITS)		/* posix semantics function */
#define FLINENO		(04<<COMBITS)		/* for/case has line number */
#define FOPTGET		(0200<<COMBITS)		/* function calls getopts */

#define TNEGATE		(01<<COMBITS)		/* ! inside [[...]] */
#define TBINARY		(02<<COMBITS)		/* binary operator in [[...]] */
#define TUNARY		(04<<COMBITS)		/* unary operator in [[...]] */
#define TTEST		(010<<COMBITS)
#define TPAREN		(TBINARY|TUNARY)
#define TSHIFT		(COMBITS+4)
#define TNSPACE		(TFUN|COMSCAN)

#define TCOM	0
#define TPAR	1
#define TFIL	2
#define TLST	3
#define TIF	4
#define TWH	5
#define TUN	(TWH|COMSCAN)
#define TTST	6
#define TSW	7
#define TAND	8
#define TORF	9
#define TFORK	10
#define TFOR	11
#define TSELECT	(TFOR|COMSCAN)
#define TARITH	12
#define	TTIME	13
#define TSETIO	14
#define TFUN	15

/* this node is a proforma for those that follow */

struct trenod
{
	int		tretyp;
	struct ionod	*treio;
};


struct forknod
{
	int		forktyp;
	struct ionod	*forkio;
	Shnode_t	*forktre;
	int		forkline;
};


struct ifnod
{
	int		iftyp;
	Shnode_t	*iftre;
	Shnode_t	*thtre;
	Shnode_t	*eltre;
};

struct whnod
{
	int		whtyp;
	Shnode_t	*whtre;
	Shnode_t	*dotre;
	struct arithnod	*whinc;
};

struct fornod
{
	int		fortyp;
	char	 	*fornam;
	Shnode_t	*fortre;
	struct comnod	*forlst;
	int		forline;
};

struct swnod
{
	int		swtyp;
	struct argnod	*swarg;
	struct regnod	*swlst;
	struct ionod	*swio;
	int		swline;
};

struct regnod
{
	struct argnod	*regptr;
	Shnode_t	*regcom;
	struct regnod	*regnxt;
	char		regflag;
};

struct parnod
{
	int		partyp;
	Shnode_t	*partre;
};

struct lstnod
{
	int		lsttyp;
	Shnode_t	*lstlef;
	Shnode_t	*lstrit;
};

/* tst is same as lst, but with extra field for line number */
struct tstnod
{
	struct lstnod	tstlst;
	int		tstline;	
};

struct functnod
{
	int		functtyp;
	char		*functnam;
	Shnode_t	*functtre;
	int		functline;
	off_t		functloc;
	struct slnod	*functstak;
	struct comnod	*functargs;
};

struct arithnod
{
	int		artyp;
	int		arline;
	struct argnod	*arexpr;
	void		*arcomp;
};


/* types of ionodes stored in iofile  */
#define IOUFD		0x3f	/* file descriptor number mask */
#define IOPUT		0x40	/* > redirection operator */
#define IOAPP		0x80	/* >> redirection operator */
#define IODOC		0x100	/* << redirection operator */
#define IOMOV		0x200	/* <& or >& operators */
#define IOCLOB		0x400	/* noclobber bit */
#define IORDW		0x800	/* <> redirection operator */
#define IORAW		0x1000	/* no expansion needed for filename */
#define IOSTRG		0x2000	/* here-document stored as incore string */
#define IOSTRIP 	0x4000	/* strip leading tabs for here-document */
#define IOQUOTE		0x8000	/* here-document delimiter was quoted */
#define IOVNM		0x10000	/* iovname field is non-zero */
#define IOLSEEK		0x20000	/* seek operators <# or >#  */
#define IOARITH		0x40000	/* arithmetic seek <# ((expr))  */
#define IOREWRITE	0x80000	/* arithmetic seek <# ((expr))  */
#define IOCOPY		IOCLOB	/* copy skipped lines onto standard output */
#define IOPROCSUB	IOARITH	/* process substitution redirection */

union Shnode_u
{
	struct argnod	arg;
	struct ionod	io;
	struct whnod	wh;
	struct swnod	sw;
	struct ifnod	if_;
	struct dolnod	dol;
	struct comnod	com;
	struct trenod	tre;
	struct forknod	fork;
	struct fornod	for_;
	struct regnod	reg;
	struct parnod	par;
	struct lstnod	lst;
	struct tstnod	tst;
	struct functnod	funct;
	struct arithnod	ar;
};

extern void			sh_freeup(Shell_t*);
extern void			sh_funstaks(struct slnod*,int);
extern Sfio_t 			*sh_subshell(Shell_t*,Shnode_t*, volatile int, int);
#if defined(__EXPORT__) && defined(_BLD_DLL) && defined(_BLD_shell) 
   __EXPORT__
#endif
extern int			sh_tdump(Sfio_t*, const Shnode_t*);
extern Shnode_t			*sh_trestore(Shell_t*, Sfio_t*);

#endif /* _SHNODES_H */
