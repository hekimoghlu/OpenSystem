/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#ifndef fcgetc
/*
 * David Korn
 * AT&T Labs
 *
 * Fast character input with sfio text streams and strings
 *
 */

#include	<sfio.h>

typedef struct _fcin
{
	Sfio_t		*_fcfile;	/* input file pointer */
	unsigned char	*fcbuff;	/* pointer to input buffer */
	unsigned char	*fclast;	/* pointer to end of input buffer */
	unsigned char	*fcptr;		/* pointer to next input char */
	unsigned char	fcchar;		/* saved character */
	short		fclen;		/* last multibyte char len */
	void (*fcfun)(Sfio_t*,const char*,int,void*);	/* advance function */
	void		*context;	/* context pointer */
	int		fcleft;		/* for multibyte boundary */
	Sfoff_t		fcoff;		/* offset for last read */
} Fcin_t;

#if SHOPT_MULTIBYTE
#   define fcmbget(x)	(mbwide()?_fcmbget(x):fcget())
#else
#   define fcmbget(x)	(fcget())
#endif
#define fcfile()	(_Fcin._fcfile)
#define fcgetc(c)	(((c=fcget()) || (c=fcfill())), c)
#define	fcget()		((int)(*_Fcin.fcptr++))
#define	fcpeek(n)	((int)_Fcin.fcptr[n])
#define	fcseek(n)	((char*)(_Fcin.fcptr+=(n)))
#define fcfirst()	((char*)_Fcin.fcbuff)
#define fclast()	((char*)_Fcin.fclast)
#define fcleft()	(_Fcin.fclast-_Fcin.fcptr)
#define fcsopen(s)	(_Fcin._fcfile=(Sfio_t*)0,_Fcin.fclen=1,_Fcin.fcbuff=_Fcin.fcptr=(unsigned char*)(s))
#define fctell()	(_Fcin.fcoff + (_Fcin.fcptr-_Fcin.fcbuff))
#define fcsave(x)	(*(x) = _Fcin)
#define fcrestore(x)	(_Fcin = *(x))
extern int		fcfill(void);
extern int		fcfopen(Sfio_t*);
extern int		fcclose(void);
void			fcnotify(void(*)(Sfio_t*,const char*,int,void*),void*);
extern int		_fcmbget(short*);

extern Fcin_t		_Fcin;		/* used by macros */

#endif /* fcgetc */
