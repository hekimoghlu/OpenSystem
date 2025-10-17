/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include <sys/cdefs.h>

#include <stdbool.h>

void	 brace_subst(char *, char **, char *, size_t);
PLAN	*find_create(char ***);
int	 find_execute(PLAN *, char **);
PLAN	*find_formplan(char **);
PLAN	*not_squish(PLAN *);
PLAN	*or_squish(PLAN *);
PLAN	*paren_squish(PLAN *);
time_t	 get_date(char *);
struct stat;
void	 printlong(char *, char *, struct stat *);
int	 queryuser(char **);
OPTION	*lookup_option(const char *);
void	 finish_execplus(void);

creat_f	c_Xmin;
creat_f	c_Xtime;
creat_f	c_acl;
creat_f	c_and;
creat_f	c_delete;
creat_f	c_depth;
creat_f	c_empty;
creat_f	c_exec;
creat_f	c_flags;
creat_f	c_follow;
creat_f	c_fstype;
creat_f	c_group;
creat_f	c_ignore_readdir_race;
creat_f	c_inum;
creat_f	c_links;
creat_f	c_ls;
creat_f	c_mXXdepth;
creat_f	c_name;
creat_f	c_newer;
creat_f	c_nogroup;
creat_f	c_nouser;
creat_f	c_perm;
creat_f	c_print;
creat_f	c_regex;
creat_f	c_samefile;
creat_f	c_simple;
creat_f	c_size;
creat_f	c_sparse;
creat_f	c_type;
creat_f	c_user;
creat_f	c_xdev;

exec_f	f_Xmin;
exec_f	f_Xtime;
exec_f	f_acl;
exec_f	f_always_true;
exec_f	f_closeparen;
exec_f	f_delete;
exec_f	f_depth;
exec_f	f_empty;
exec_f	f_exec;
exec_f	f_expr;
exec_f	f_false;
exec_f	f_flags;
exec_f	f_fstype;
exec_f	f_group;
exec_f	f_inum;
exec_f	f_links;
exec_f	f_ls;
exec_f	f_name;
exec_f	f_newer;
exec_f	f_nogroup;
exec_f	f_not;
exec_f	f_nouser;
exec_f	f_openparen;
exec_f	f_or;
exec_f	f_path;
exec_f	f_perm;
exec_f	f_print;
exec_f	f_print0;
exec_f	f_prune;
exec_f	f_quit;
exec_f	f_regex;
exec_f	f_size;
exec_f	f_sparse;
exec_f	f_type;
exec_f	f_user;
#ifdef __APPLE__
exec_f	f_xattr;
exec_f	f_xattrname;
#endif /* __APPLE__ */

extern int ftsoptions, ignore_readdir_race, isdepth, isoutput;
extern int issort, isxargs;
extern int mindepth, maxdepth;
extern int regexp_flags;
extern int exitstatus;
extern time_t now;
extern int dotfd;
extern FTS *tree;

#ifdef __APPLE__
extern bool unix2003_compat;
#else
#define	unix2003_compat	true
#endif
