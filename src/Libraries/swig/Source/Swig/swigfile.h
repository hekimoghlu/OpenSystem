/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
/* $Id: swig.h 9603 2006-12-05 21:47:01Z beazley $ */

extern List   *Swig_add_directory(const_String_or_char_ptr dirname);
extern void    Swig_push_directory(const_String_or_char_ptr dirname);
extern void    Swig_pop_directory(void);
extern String *Swig_last_file(void);
extern List   *Swig_search_path(void);
extern FILE   *Swig_include_open(const_String_or_char_ptr name);
extern FILE   *Swig_open(const_String_or_char_ptr name);
extern String *Swig_read_file(FILE *f); 
extern String *Swig_include(const_String_or_char_ptr name);
extern String *Swig_include_sys(const_String_or_char_ptr name);
extern int     Swig_insert_file(const_String_or_char_ptr name, File *outfile);
extern void    Swig_set_push_dir(int dopush);
extern int     Swig_get_push_dir(void);
extern void    Swig_register_filebyname(const_String_or_char_ptr filename, File *outfile);
extern File   *Swig_filebyname(const_String_or_char_ptr filename);
extern char   *Swig_file_suffix(const_String_or_char_ptr filename);
extern char   *Swig_file_basename(const_String_or_char_ptr filename);
extern char   *Swig_file_filename(const_String_or_char_ptr filename);
extern char   *Swig_file_dirname(const_String_or_char_ptr filename);

/* Delimiter used in accessing files and directories */

#if defined(MACSWIG)
#  define SWIG_FILE_DELIMITER ":"
#elif defined(_WIN32)
#  define SWIG_FILE_DELIMITER "\\"
#else
#  define SWIG_FILE_DELIMITER "/"
#endif
