/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#ifndef _SFIO_T_H
#define _SFIO_T_H	1

/*	This header file is for library writers who need to know certain
**	internal info concerning the full Sfio_t structure. Including this
**	file means that you agree to track closely with sfio development
**	in case its internal architecture is changed.
**
**	Written by Kiem-Phong Vo
*/

/* the parts of Sfio_t private to sfio functions */
#define _SFIO_PRIVATE \
	Sfoff_t			extent;	/* current file	size		*/ \
	Sfoff_t			here;	/* current physical location	*/ \
	unsigned char		unused_1;/* unused #1			*/ \
	unsigned char		tiny[1];/* for unbuffered read stream	*/ \
	unsigned short		bits;	/* private flags		*/ \
	unsigned int		mode;	/* current io mode		*/ \
	struct _sfdisc_s*	disc;	/* discipline			*/ \
	struct _sfpool_s*	pool;	/* the pool containing this	*/ \
	struct _sfrsrv_s*	rsrv;	/* reserved buffer		*/ \
	struct _sfproc_s*	proc;	/* coprocess id, etc.		*/ \
	Void_t*			mutex;	/* mutex for thread-safety	*/ \
	Void_t*			stdio;	/* stdio FILE if any		*/ \
	Sfoff_t			lpos;	/* last seek position		*/ \
	size_t			iosz;	/* preferred size for I/O	*/ \
	size_t			blksz;	/* preferred block size		*/ \
	int			getr;	/* the last sfgetr separator 	*/ \
	_SFIO_PRIVATE_PAD

#if _ast_sizeof_pointer == 8
#define _SFIO_PRIVATE_PAD	int pad;
#else
#define _SFIO_PRIVATE_PAD
#endif

#include	"sfio.h"

/* mode bit to indicate that the structure hasn't been initialized */
#define SF_INIT		0000004
#define SF_DCDOWN	00010000

/* short-hand for common stream types */
#define SF_RDWR		(SF_READ|SF_WRITE)
#define SF_RDSTR	(SF_READ|SF_STRING)
#define SF_WRSTR	(SF_WRITE|SF_STRING)
#define SF_RDWRSTR	(SF_RDWR|SF_STRING)

/* for static initialization of an Sfio_t structure */
#define SFNEW(data,size,file,type,disc,mutex)	\
	{ (unsigned char*)(data),			/* next		*/ \
	  (unsigned char*)(data),			/* endw		*/ \
	  (unsigned char*)(data),			/* endr		*/ \
	  (unsigned char*)(data),			/* endb		*/ \
	  (Sfio_t*)0,					/* push		*/ \
	  (unsigned short)((type)&SF_FLAGS),		/* flags	*/ \
	  (short)(file),				/* file		*/ \
	  (unsigned char*)(data),			/* data		*/ \
	  (ssize_t)(size),				/* size		*/ \
	  (ssize_t)(-1),				/* val		*/ \
	  (Sfoff_t)0,					/* extent	*/ \
	  (Sfoff_t)0,					/* here		*/ \
	  0,						/* getr		*/ \
	  {0},						/* tiny		*/ \
	  0,						/* bits		*/ \
	  (unsigned int)(((type)&(SF_RDWR))|SF_INIT),	/* mode		*/ \
	  (struct _sfdisc_s*)(disc),			/* disc		*/ \
	  (struct _sfpool_s*)0,				/* pool		*/ \
	  (struct _sfrsrv_s*)0,				/* rsrv		*/ \
	  (struct _sfproc_s*)0,				/* proc		*/ \
	  (mutex),					/* mutex	*/ \
	  (Void_t*)0,					/* stdio	*/ \
	  (Sfoff_t)0,					/* lpos		*/ \
	  (size_t)0					/* iosz		*/ \
	}

/* function to clear an Sfio_t structure */
#define SFCLEAR(f,mtx) \
	( (f)->next = (unsigned char*)0,		/* next		*/ \
	  (f)->endw = (unsigned char*)0,		/* endw		*/ \
	  (f)->endr = (unsigned char*)0,		/* endr		*/ \
	  (f)->endb = (unsigned char*)0,		/* endb		*/ \
	  (f)->push = (Sfio_t*)0,			/* push		*/ \
	  (f)->flags = (unsigned short)0,		/* flags	*/ \
	  (f)->file = -1,				/* file		*/ \
	  (f)->data = (unsigned char*)0,		/* data		*/ \
	  (f)->size = (ssize_t)(-1),			/* size		*/ \
	  (f)->val = (ssize_t)(-1),			/* val		*/ \
	  (f)->extent = (Sfoff_t)(-1),			/* extent	*/ \
	  (f)->here = (Sfoff_t)0,			/* here		*/ \
	  (f)->getr = 0,				/* getr		*/ \
	  (f)->tiny[0] = 0,				/* tiny		*/ \
	  (f)->bits = 0,				/* bits		*/ \
	  (f)->mode = 0,				/* mode		*/ \
	  (f)->disc = (struct _sfdisc_s*)0,		/* disc		*/ \
	  (f)->pool = (struct _sfpool_s*)0,		/* pool		*/ \
	  (f)->rsrv = (struct _sfrsrv_s*)0,		/* rsrv		*/ \
	  (f)->proc = (struct _sfproc_s*)0,		/* proc		*/ \
	  (f)->mutex = (mtx),				/* mutex	*/ \
	  (f)->stdio = (Void_t*)0,			/* stdio	*/ \
	  (f)->lpos = (Sfoff_t)0,			/* lpos		*/ \
	  (f)->iosz = (size_t)0				/* iosz		*/ \
	)

/* expose next stream inside discipline function; state saved in int f */
#define SFDCNEXT(sp,f)	(((f)=(sp)->bits&SF_DCDOWN),(sp)->bits|=SF_DCDOWN)

/* restore SFDCNEXT() state from int f */
#define SFDCPREV(sp,f)	((f)?(0):((sp)->bits&=~SF_DCDOWN))

#endif /* _SFIO_T_H */
