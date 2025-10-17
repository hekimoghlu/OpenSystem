/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#if !defined(ATF_C_FS_H)
#define ATF_C_FS_H

#include <sys/types.h>
#include <sys/stat.h>

#include <stdarg.h>
#include <stdbool.h>

#include <atf-c/error_fwd.h>

#include "dynstr.h"

/* ---------------------------------------------------------------------
 * The "atf_fs_path" type.
 * --------------------------------------------------------------------- */

struct atf_fs_path {
    atf_dynstr_t m_data;
};
typedef struct atf_fs_path atf_fs_path_t;

/* Constructors/destructors. */
atf_error_t atf_fs_path_init_ap(atf_fs_path_t *, const char *, va_list);
atf_error_t atf_fs_path_init_fmt(atf_fs_path_t *, const char *, ...);
atf_error_t atf_fs_path_copy(atf_fs_path_t *, const atf_fs_path_t *);
void atf_fs_path_fini(atf_fs_path_t *);

/* Getters. */
atf_error_t atf_fs_path_branch_path(const atf_fs_path_t *, atf_fs_path_t *);
const char *atf_fs_path_cstring(const atf_fs_path_t *);
atf_error_t atf_fs_path_leaf_name(const atf_fs_path_t *, atf_dynstr_t *);
bool atf_fs_path_is_absolute(const atf_fs_path_t *);
bool atf_fs_path_is_root(const atf_fs_path_t *);

/* Modifiers. */
atf_error_t atf_fs_path_append_ap(atf_fs_path_t *, const char *, va_list);
atf_error_t atf_fs_path_append_fmt(atf_fs_path_t *, const char *, ...);
atf_error_t atf_fs_path_append_path(atf_fs_path_t *, const atf_fs_path_t *);
atf_error_t atf_fs_path_to_absolute(const atf_fs_path_t *, atf_fs_path_t *);

/* Operators. */
bool atf_equal_fs_path_fs_path(const atf_fs_path_t *,
                               const atf_fs_path_t *);

/* ---------------------------------------------------------------------
 * The "atf_fs_stat" type.
 * --------------------------------------------------------------------- */

struct atf_fs_stat {
    int m_type;
    struct stat m_sb;
};
typedef struct atf_fs_stat atf_fs_stat_t;

/* Constants. */
extern const int atf_fs_stat_blk_type;
extern const int atf_fs_stat_chr_type;
extern const int atf_fs_stat_dir_type;
extern const int atf_fs_stat_fifo_type;
extern const int atf_fs_stat_lnk_type;
extern const int atf_fs_stat_reg_type;
extern const int atf_fs_stat_sock_type;
extern const int atf_fs_stat_wht_type;

/* Constructors/destructors. */
atf_error_t atf_fs_stat_init(atf_fs_stat_t *, const atf_fs_path_t *);
void atf_fs_stat_copy(atf_fs_stat_t *, const atf_fs_stat_t *);
void atf_fs_stat_fini(atf_fs_stat_t *);

/* Getters. */
dev_t atf_fs_stat_get_device(const atf_fs_stat_t *);
ino_t atf_fs_stat_get_inode(const atf_fs_stat_t *);
mode_t atf_fs_stat_get_mode(const atf_fs_stat_t *);
off_t atf_fs_stat_get_size(const atf_fs_stat_t *);
int atf_fs_stat_get_type(const atf_fs_stat_t *);
bool atf_fs_stat_is_owner_readable(const atf_fs_stat_t *);
bool atf_fs_stat_is_owner_writable(const atf_fs_stat_t *);
bool atf_fs_stat_is_owner_executable(const atf_fs_stat_t *);
bool atf_fs_stat_is_group_readable(const atf_fs_stat_t *);
bool atf_fs_stat_is_group_writable(const atf_fs_stat_t *);
bool atf_fs_stat_is_group_executable(const atf_fs_stat_t *);
bool atf_fs_stat_is_other_readable(const atf_fs_stat_t *);
bool atf_fs_stat_is_other_writable(const atf_fs_stat_t *);
bool atf_fs_stat_is_other_executable(const atf_fs_stat_t *);

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

extern const int atf_fs_access_f;
extern const int atf_fs_access_r;
extern const int atf_fs_access_w;
extern const int atf_fs_access_x;

atf_error_t atf_fs_eaccess(const atf_fs_path_t *, int);
atf_error_t atf_fs_exists(const atf_fs_path_t *, bool *);
atf_error_t atf_fs_getcwd(atf_fs_path_t *);
atf_error_t atf_fs_mkdtemp(atf_fs_path_t *);
atf_error_t atf_fs_mkstemp(atf_fs_path_t *, int *);
atf_error_t atf_fs_rmdir(const atf_fs_path_t *);
atf_error_t atf_fs_unlink(const atf_fs_path_t *);

#endif /* !defined(ATF_C_FS_H) */
