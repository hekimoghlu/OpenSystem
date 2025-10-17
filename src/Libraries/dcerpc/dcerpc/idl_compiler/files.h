/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
**
**  NAME:
**
**      files.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Header file for file manipulation routines.
**
**  VERSION: DCE 1.0
**
*/

#ifndef files_incl
#define files_incl

#ifndef S_IFREG
#  include <sys/types.h>
#  include <sys/stat.h>
#endif

#include <nidl.h>               /* IDL common defs */
#include <nametbl.h>

typedef enum                    /* Filespec kinds: */
{
    file_dir,                   /* Directory */
    file_file,                  /* Regular ol' file */
    file_special                /* Something else */
} FILE_k_t;

extern boolean FILE_open(
    char *filespec,
    FILE **fid
);

extern boolean FILE_create(
    char *filespec,
    FILE **fid
);

extern boolean FILE_lookup(
    char const  *filespec,
    char const  * const *idir_list,
    struct stat *stat_buf,
    char        *lookup_spec,
	size_t		lookup_spec_len
);

extern boolean FILE_form_filespec(
    char const *in_filespec,
    char const *dir,
    char const *type,
    char const *rel_filespec,
    char       *out_filespec,
	size_t     out_filespec_len
);

extern boolean FILE_parse(
    char const *filespec,
    char       *dir,
	size_t		dir_len,
    char       *name,
	size_t		name_len,
    char       *type,
	size_t		type_len
);

extern boolean FILE_has_dir_info(
    char const *filespec
);

extern boolean FILE_is_cwd(
    char *filespec
);

extern boolean FILE_kind(
    char const  *filespec,
    FILE_k_t    *filekind
);

extern int FILE_execute_cmd(
    char        *cmd_string,
    char        *p1,
    char        *p2,
    long        msg_id
);

extern void FILE_delete(
    char        *filename
);

#endif /* files_incl */
