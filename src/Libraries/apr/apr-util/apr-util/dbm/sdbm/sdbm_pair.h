/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#ifndef SDBM_PAIR_H
#define SDBM_PAIR_H

/* Mini EMBED (pair.c) */
#define chkpage apu__sdbm_chkpage
#define delpair apu__sdbm_delpair
#define duppair apu__sdbm_duppair
#define fitpair apu__sdbm_fitpair
#define getnkey apu__sdbm_getnkey
#define getpair apu__sdbm_getpair
#define putpair apu__sdbm_putpair
#define splpage apu__sdbm_splpage

int fitpair(char *, int);
void  putpair(char *, apr_sdbm_datum_t, apr_sdbm_datum_t);
apr_sdbm_datum_t getpair(char *, apr_sdbm_datum_t);
int  delpair(char *, apr_sdbm_datum_t);
int  chkpage (char *);
apr_sdbm_datum_t getnkey(char *, int);
void splpage(char *, char *, long);
int duppair(char *, apr_sdbm_datum_t);

#endif /* SDBM_PAIR_H */

