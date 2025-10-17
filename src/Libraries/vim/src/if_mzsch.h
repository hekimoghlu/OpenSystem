/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#ifndef _IF_MZSCH_H_
#define _IF_MZSCH_H_
#ifdef __MINGW32__
// Hack to engage Cygwin-specific settings
# define __CYGWIN32__
# include <stdint.h>
#endif

#ifdef PROTO
// avoid syntax error for defining Thread_Local_Variables.
# define __thread // empty
#endif

// #ifdef needed for "make depend"
#ifdef FEAT_MZSCHEME
# include <schvers.h>
# include <scheme.h>
#endif

#ifdef __MINGW32__
# undef __CYGWIN32__
#endif

#if MZSCHEME_VERSION_MAJOR >= 299
# define SCHEME_STRINGP(obj) (SCHEME_BYTE_STRINGP(obj) || SCHEME_CHAR_STRINGP(obj))
# define BYTE_STRING_VALUE(obj) ((char_u *)SCHEME_BYTE_STR_VAL(obj))
#else
// macros for compatibility with older versions
# define scheme_current_config() scheme_config
# define scheme_make_sized_byte_string scheme_make_sized_string
# define scheme_format_utf8 scheme_format
# ifndef DYNAMIC_MZSCHEME
// for dynamic MzScheme there will be separate definitions in if_mzsch.c
#  define scheme_get_sized_byte_string_output scheme_get_sized_string_output
#  define scheme_make_byte_string scheme_make_string
#  define scheme_make_byte_string_output_port scheme_make_string_output_port
# endif

# define SCHEME_BYTE_STRLEN_VAL SCHEME_STRLEN_VAL
# define BYTE_STRING_VALUE(obj) ((char_u *)SCHEME_STR_VAL(obj))
# define scheme_byte_string_to_char_string(obj) (obj)
# define SCHEME_BYTE_STRINGP SCHEME_STRINGP
#endif

// Precise GC macros
#ifndef MZ_GC_DECL_REG
# define MZ_GC_DECL_REG(size)		 // empty
#endif
#ifndef MZ_GC_VAR_IN_REG
# define MZ_GC_VAR_IN_REG(x, v)		 // empty
#endif
#ifndef MZ_GC_ARRAY_VAR_IN_REG
# define MZ_GC_ARRAY_VAR_IN_REG(x, v, l) // empty
#endif
#ifndef MZ_GC_REG
# define MZ_GC_REG()			 // empty
#endif
#ifndef MZ_GC_UNREG
# define MZ_GC_UNREG()			 // empty
#endif

#ifdef MZSCHEME_FORCE_GC
/*
 * force garbage collection to check all references are registered
 * seg faults will indicate not registered refs
 */
# define MZ_GC_CHECK() scheme_collect_garbage();
#else
# define MZ_GC_CHECK()			// empty
#endif

#endif // _IF_MZSCH_H_
