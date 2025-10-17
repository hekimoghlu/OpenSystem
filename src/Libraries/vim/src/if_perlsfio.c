/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
 * if_perlsfio.c: Special I/O functions for Perl interface.
 */

#define _memory_h	// avoid memset redeclaration
#define IN_PERL_FILE	// don't include if_perl.pro from prot.h

#include "vim.h"

#if defined(USE_SFIO) || defined(PROTO)

#ifndef USE_SFIO	// just generating prototypes
# define Sfio_t int
# define Sfdisc_t int
#endif

#define NIL(type)	((type)0)

    static int
sfvimwrite(
    Sfio_t	    *f,		// stream involved
    char	    *buf,	// buffer to read from
    int		    n,		// number of bytes to write
    Sfdisc_t	    *disc)	// discipline
{
    char_u *str;

    str = vim_strnsave((char_u *)buf, n);
    if (str == NULL)
	return 0;
    msg_split((char *)str);
    vim_free(str);

    return n;
}

/*
 * sfdcnewnvi --
 *  Create Vim discipline
 */
    Sfdisc_t *
sfdcnewvim(void)
{
    Sfdisc_t	*disc;

    disc = ALLOC_ONE(Sfdisc_t);
    if (disc == NULL)
	return NULL;

    disc->readf = (Sfread_f)NULL;
    disc->writef = sfvimwrite;
    disc->seekf = (Sfseek_f)NULL;
    disc->exceptf = (Sfexcept_f)NULL;

    return disc;
}

#endif // USE_SFIO
