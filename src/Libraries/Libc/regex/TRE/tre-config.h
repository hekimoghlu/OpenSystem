/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
/* tre-config.h.in.  This file has all definitions that are needed in
   `tre.h'.  Note that this file must contain only the bare minimum
   of definitions without the TRE_ prefix to avoid conflicts between
   definitions here and definitions included from somewhere else. */

/* Define to 1 if you have the <libutf8.h> header file. */
/* #undef HAVE_LIBUTF8_H */

/* Define to 1 if the system has the type `reg_errcode_t'. */
/* #undef HAVE_REG_ERRCODE_T */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <wchar.h> header file. */
#define HAVE_WCHAR_H 1

/* Define if you want to enable approximate matching functionality. */
/* #undef TRE_APPROX */

/* Define to enable multibyte character set support. */
#define TRE_MULTIBYTE 1

/* Define to the absolute path to the system tre.h */
#define TRE_SYSTEM_REGEX_H_PATH <regex.h>

/* Define to include the system regex.h from tre.h */
#define TRE_USE_SYSTEM_REGEX_H 1

/* Define to enable wide character (wchar_t) support. */
#define TRE_WCHAR 1

/* TRE version string. */
#define TRE_VERSION "0.8.0"

/* TRE version level 1. */
#define TRE_VERSION_1 0

/* TRE version level 2. */
#define TRE_VERSION_2 8

/* TRE version level 3. */
#define TRE_VERSION_3 0
