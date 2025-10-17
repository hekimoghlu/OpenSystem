/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
#ifndef YASM_NASMLIB_H
#define YASM_NASMLIB_H

/*
 * Wrappers around malloc, realloc and free. nasm_malloc will
 * fatal-error and die rather than return NULL; nasm_realloc will
 * do likewise, and will also guarantee to work right on being
 * passed a NULL pointer; nasm_free will do nothing if it is passed
 * a NULL pointer.
 */
#define nasm_malloc yasm_xmalloc
#define nasm_realloc yasm_xrealloc
#ifdef WITH_DMALLOC
#define nasm_free(p) do { if (p) yasm_xfree(p); } while(0)
#else
#define nasm_free(p) yasm_xfree(p)
#endif
#define nasm_strdup yasm__xstrdup
#define nasm_strndup yasm__xstrndup
#define nasm_stricmp yasm__strcasecmp
#define nasm_strnicmp yasm__strncasecmp

/*
 * Convert a string into a number, using NASM number rules. Sets
 * `*error' to TRUE if an error occurs, and FALSE otherwise.
 */
yasm_intnum *nasm_readnum(char *str, int *error);

/*
 * Convert a character constant into a number. Sets
 * `*warn' to TRUE if an overflow occurs, and FALSE otherwise.
 * str points to and length covers the middle of the string,
 * without the quotes.
 */
yasm_intnum *nasm_readstrnum(char *str, size_t length, int *warn);

char *nasm_src_set_fname(char *newname);
char *nasm_src_get_fname(void);
long nasm_src_set_linnum(long newline);
long nasm_src_get_linnum(void);
/*
 * src_get may be used if you simply want to know the source file and line.
 * It is also used if you maintain private status about the source location
 * It return 0 if the information was the same as the last time you
 * checked, -1 if the name changed and (new-old) if just the line changed.
 */
int nasm_src_get(long *xline, char **xname);

void nasm_quote(char **str);
char *nasm_strcat(const char *one, const char *two);

#endif
