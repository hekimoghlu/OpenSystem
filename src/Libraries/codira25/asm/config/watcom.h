/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
 * config/watcom.h
 *
 * Compiler definitions for OpenWatcom instead of config.h.in.
 *  See config.h.in for the variables which can be defined here.
 *
 * This was taken from openwcom.mak and needs to be actually validated.
 */

#ifndef NASM_CONFIG_WATCOM_H
#define NASM_CONFIG_WATCOM_H

#define HAVE_DECL_STRCASECMP 1
#define HAVE_DECL_STRICMP 1
#define HAVE_DECL_STRLCPY 1
#define HAVE_DECL_STRNCASECMP 1
#define HAVE_DECL_STRNICMP 1
#ifndef __LINUX__
#define HAVE_IO_H 1
#endif
#define HAVE_LIMITS_H 1
#define HAVE_MEMORY_H 1
#define HAVE_SNPRINTF 1
#if (__WATCOMC__ >= 1230)
#undef HAVE__BOOL /* need stdbool.h */
#define HAVE_STDBOOL_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_STDINT_H 1
#define HAVE_UINTPTR_T 1
#endif
#define HAVE_STDLIB_H 1
#define HAVE_STRCSPN 1
#define HAVE_STRICMP 1
#define HAVE_STRNICMP 1
#define HAVE_STRSPN 1
#define HAVE_STRING_H 1
#if (__WATCOMC__ >= 1240)
#define HAVE_STRCASECMP 1
#define HAVE_STRNCASECMP 1
#define HAVE_STRLCPY 1
#define HAVE_STRINGS_H 1
#endif
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_FCNTL_H 1
#define HAVE_UNISTD_H 1
#define HAVE_VSNPRINTF 1
#define STDC_HEADERS 1

#define HAVE__FULLPATH 1
#define HAVE_ACCESS
#define HAVE_STRUCT_STAT
#define HAVE_STAT
#define HAVE_FSTAT
#define HAVE_FILENO
#ifdef __LINUX__
#define HAVE_FTRUNCATE
#else
#define HAVE_CHSIZE
#define HAVE__CHSIZE
#endif
#define HAVE_ISASCII
#define HAVE_ISCNTRL

#if (__WATCOMC__ >= 1250)
#define restrict __restrict
#else
#define restrict
#endif
#define inline __inline

#endif /* NASM_CONFIG_WATCOM_H */
