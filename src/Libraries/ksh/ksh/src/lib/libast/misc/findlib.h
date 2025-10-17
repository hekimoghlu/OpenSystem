/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
 * fast find private interface
 */

#ifndef _FINDLIB_H
#define _FINDLIB_H

#include <ast.h>
#include <cdt.h>
#include <ctype.h>
#include <error.h>
#include <ls.h>
#include <regex.h>
#include <vmalloc.h>

#define FF_old		1	/* old format - 7 bit bigram		*/
#define FF_gnu		2	/* gnu 8 bit no bigram			*/
#define FF_dir		3	/* FF_gnu, dirs have trailing /		*/
#define FF_typ		4	/* FF_dir with types			*/

#define FF_gnu_magic	"LOCATE02"
#define FF_dir_magic	"FIND-DIR-02"
#define FF_typ_magic	"FIND-DIR-TYPE-03"

#define FF_ESC		0036
#define FF_MAX		0200
#define FF_MIN		0040
#define FF_OFF		0016

#define FF_SET_TYPE(p,i)	((p)->decode.bigram1[((i)>>3)&((1<<CHAR_BIT)-1)]|=(1<<((i)&07)))
#define FF_OK_TYPE(p,i)		(!(p)->types||((p)->decode.bigram1[((i)>>3)&((1<<CHAR_BIT)-1)]&(1<<((i)&07))))

typedef struct
{
	char*		end;
	char*		type;
	char*		restore;
	int		count;
	int		found;
	int		ignorecase;
	int		match;
	int		peek;
	int		swap;
	regex_t		re;
	char		bigram1[(1<<(CHAR_BIT-1))];
	char		bigram2[(1<<(CHAR_BIT-1))];
	char		path[PATH_MAX];
	char		temp[PATH_MAX];
	char		pattern[1];
} Decode_t;

typedef struct
{
	Dtdisc_t	namedisc;
	Dtdisc_t	indexdisc;
	Dt_t*		namedict;
	Dt_t*		indexdict;
	int		prefix;
	unsigned char	bigram[2*FF_MAX];
	unsigned short	code[FF_MAX][FF_MAX];
	unsigned short	hits[USHRT_MAX+1];
	char		path[PATH_MAX];
	char		mark[PATH_MAX];
	char		file[PATH_MAX];
	char		temp[PATH_MAX];
} Encode_t;

typedef union
{
	Decode_t	code_decode;
	Encode_t	code_encode;
} Code_t;

typedef struct
{
	Dtlink_t	byname;
	Dtlink_t	byindex;
	unsigned long	index;
	char		name[1];
} Type_t;

#define _FIND_PRIVATE_			\
	Finddisc_t*	disc;		\
	Vmalloc_t*	vm;		\
	char**		dirs;		\
	int*		lens;		\
	Sfio_t*		fp;		\
	Findverify_f	verifyf;	\
	int		generate;	\
	int		method;		\
	int		secure;		\
	int		types;		\
	int		verify;		\
	Code_t		code;

#define decode		code.code_decode
#define encode		code.code_encode

#include <find.h>

#endif
