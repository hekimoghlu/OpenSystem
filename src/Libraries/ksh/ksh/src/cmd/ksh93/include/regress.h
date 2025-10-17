/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
 * David Korn
 * AT&T Labs
 *
 * Shell interface private definitions
 *
 */

#ifndef _REGRESS_H
#define _REGRESS_H		1

#if SHOPT_REGRESS

typedef struct Regress_s
{
	Shopt_t	options;
} Regress_t;

#define sh_isregress(r)		is_option(&sh.regress->options,r)
#define sh_onregress(r)		on_option(&sh.regress->options,r)
#define sh_offregress(r)	off_option(&sh.regress->options,r)

#define REGRESS(r,i,f)		do { if (sh_isregress(REGRESS_##r)) sh_regress(REGRESS_##r, i, sfprints f, __LINE__, __FILE__); } while (0)

#define REGRESS_egid		1
#define REGRESS_euid		2
#define REGRESS_p_suid		3
#define REGRESS_source		4
#define REGRESS_etc		5

#undef	SHOPT_P_SUID
#define SHOPT_P_SUID		sh_regress_p_suid(__LINE__, __FILE__)

extern int			b___regress__(int, char**, Shbltin_t*);
extern void			sh_regress_init(Shell_t*);
extern void			sh_regress(unsigned int, const char*, const char*, unsigned int, const char*);
extern uid_t			sh_regress_p_suid(unsigned int, const char*);
extern char*			sh_regress_etc(const char*, unsigned int, const char*);

#else

#define REGRESS(r,i,f)

#endif /* SHOPT_REGRESS */

#endif /* _REGRESS_H */
