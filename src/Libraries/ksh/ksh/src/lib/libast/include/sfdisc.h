/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
 * AT&T Research
 *
 * sfio discipline interface definitions
 */

#ifndef _SFDISC_H
#define _SFDISC_H

#include <ast.h>

#define SFDCEVENT(a,b,n)	((((a)-'A'+1)<<11)^(((b)-'A'+1)<<6)^(n))

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

#define SFSK_DISCARD		SFDCEVENT('S','K',1)

/*
 * %(...) printf support
 */

typedef int (*Sf_key_lookup_t)(void*, Sffmt_t*, const char*, char**, Sflong_t*);
typedef char* (*Sf_key_convert_t)(void*, Sffmt_t*, const char*, char*, Sflong_t);

extern int		sfkeyprintf(Sfio_t*, void*, const char*, Sf_key_lookup_t, Sf_key_convert_t);
extern int		sfkeyprintf_20000308(Sfio_t*, void*, const char*, Sf_key_lookup_t, Sf_key_convert_t);

/*
 * pure sfio read and/or write disciplines
 */

extern int		sfdcdio(Sfio_t*, size_t);
extern int		sfdcdos(Sfio_t*);
extern int		sfdcfilter(Sfio_t*, const char*);
extern int		sfdcmore(Sfio_t*, const char*, int, int);
extern int		sfdcprefix(Sfio_t*, const char*);
extern int		sfdcseekable(Sfio_t*);
extern int		sfdcslow(Sfio_t*);
extern int		sfdctee(Sfio_t*, Sfio_t*);
extern int		sfdcunion(Sfio_t*, Sfio_t**, int);

extern Sfio_t*		sfdcsubstream(Sfio_t*, Sfio_t*, Sfoff_t, Sfoff_t);

#undef	extern

#endif
