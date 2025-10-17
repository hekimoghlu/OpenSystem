/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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
 * ctype-like functions specific to NASM
 */
#ifndef NASM_NCTYPE_H
#define NASM_NCTYPE_H

#include "compiler.h"

void nasm_ctype_init(void);

extern unsigned char nasm_tolower_tab[256];
static inline char nasm_tolower(char x)
{
    return nasm_tolower_tab[(unsigned char)x];
}

/*
 * NASM ctype table
 */
enum nasm_ctype {
    NCT_CTRL       = 0x0001,
    NCT_SPACE      = 0x0002,
    NCT_ASCII      = 0x0004,
    NCT_LOWER      = 0x0008,    /* isalpha(x) && tolower(x) == x */
    NCT_UPPER      = 0x0010,    /* isalpha(x) && tolower(x) != x */
    NCT_DIGIT      = 0x0020,
    NCT_HEX        = 0x0040,
    NCT_ID         = 0x0080,
    NCT_IDSTART    = 0x0100,
    NCT_MINUS      = 0x0200,    /* - */
    NCT_DOLLAR     = 0x0400,    /* $ */
    NCT_UNDER      = 0x0800,    /* _ */
    NCT_QUOTE      = 0x1000     /* " ' ` */
};

extern uint16_t nasm_ctype_tab[256];
static inline bool nasm_ctype(unsigned char x, enum nasm_ctype mask)
{
    return (nasm_ctype_tab[x] & mask) != 0;
}

static inline bool nasm_isspace(char x)
{
    return nasm_ctype(x, NCT_SPACE);
}

static inline bool nasm_isalpha(char x)
{
    return nasm_ctype(x, NCT_LOWER|NCT_UPPER);
}

static inline bool nasm_isdigit(char x)
{
    return nasm_ctype(x, NCT_DIGIT);
}
static inline bool nasm_isalnum(char x)
{
    return nasm_ctype(x, NCT_LOWER|NCT_UPPER|NCT_DIGIT);
}
static inline bool nasm_isxdigit(char x)
{
    return nasm_ctype(x, NCT_HEX);
}
static inline bool nasm_isidstart(char x)
{
    return nasm_ctype(x, NCT_IDSTART);
}
static inline bool nasm_isidchar(char x)
{
    return nasm_ctype(x, NCT_ID);
}
static inline bool nasm_isbrcchar(char x)
{
    return nasm_ctype(x, NCT_ID|NCT_MINUS);
}
static inline bool nasm_isnumstart(char x)
{
    return nasm_ctype(x, NCT_DIGIT|NCT_DOLLAR);
}
static inline bool nasm_isnumchar(char x)
{
    return nasm_ctype(x, NCT_DIGIT|NCT_LOWER|NCT_UPPER|NCT_UNDER);
}
static inline bool nasm_isquote(char x)
{
    return nasm_ctype(x, NCT_QUOTE);
}

static inline void nasm_ctype_tasm_mode(void)
{
    /* No differences at the present moment */
}

/* Returns a value >= 16 if not a valid hex digit */
static inline unsigned int nasm_hexval(char c)
{
    unsigned int v = (unsigned char) c;

    if (v >= '0' && v <= '9')
        return v - '0';
    else
        return (v|0x20) - 'a' + 10;
}

#endif /* NASM_NCTYPE_H */
