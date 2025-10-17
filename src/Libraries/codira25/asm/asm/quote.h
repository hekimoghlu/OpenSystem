/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#ifndef NASM_QUOTE_H
#define NASM_QUOTE_H

#include "compiler.h"

char *nasm_quote(const char *str, size_t *len);
char *nasm_quote_cstr(const char *str, size_t *len);
size_t nasm_unquote_anystr(char *str, char **endptr,
                           uint32_t badctl, char qstart);
size_t nasm_unquote(char *str, char **endptr);
size_t nasm_unquote_cstr(char *str, char **endptr);
char *nasm_skip_string(const char *str);

/* Arguments used with nasm_quote_anystr() */

/*
 * These are the only control characters when we produce a C string:
 * BEL BS TAB ESC
 */
#define OKCTL ((1U << '\a') | (1U << '\b') | (1U << '\t') | (1U << 27))
#define BADCTL (~(uint32_t)OKCTL)

/* Initial quotation mark */
#define STR_C    '\"'
#define STR_NASM '`'

#endif /* NASM_QUOTE_H */

