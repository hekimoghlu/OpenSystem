/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#ifndef SUDO_EDIT_H
#define SUDO_EDIT_H

/*
 * Directory open flags for use with openat(2).
 * Use O_SEARCH/O_PATH and/or O_DIRECTORY where possible.
 */
#if defined(O_SEARCH)
# if defined(O_DIRECTORY)
#  define DIR_OPEN_FLAGS	(O_SEARCH|O_DIRECTORY)
# else
#  define DIR_OPEN_FLAGS	(O_SEARCH)
# endif
#elif defined(O_PATH)
# if defined(O_DIRECTORY)
#  define DIR_OPEN_FLAGS	(O_PATH|O_DIRECTORY)
# else
#  define DIR_OPEN_FLAGS	(O_PATH)
# endif
#elif defined(O_DIRECTORY)
# define DIR_OPEN_FLAGS		(O_RDONLY|O_DIRECTORY)
#else
# define DIR_OPEN_FLAGS		(O_RDONLY|O_NONBLOCK)
#endif

/* copy_file.c */
int sudo_copy_file(const char *src, int src_fd, off_t src_len, const char *dst, int dst_fd, off_t dst_len);
bool sudo_check_temp_file(int tfd, const char *tname, uid_t uid, struct stat *sb);

/* edit_open.c */
struct sudo_cred;
void switch_user(uid_t euid, gid_t egid, int ngroups, GETGROUPS_T *groups);
int sudo_edit_open(char *path, int oflags, mode_t mode, int sflags, struct sudo_cred *user_cred, struct sudo_cred *cur_cred);
int dir_is_writable(int dfd, struct sudo_cred *user_cred, struct sudo_cred *cur_cred);
bool sudo_edit_parent_valid(char *path, int sflags, struct sudo_cred *user_cred, struct sudo_cred *cur_cred);

#endif /* SUDO_EDIT_H */
