/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#ifndef _CP_EXTERN_H_
#define _CP_EXTERN_H_

typedef struct {
	char	*p_end;			/* pointer to NULL at end of path */
	char	*target_end;		/* pointer to end of target base */
	char	p_path[PATH_MAX];	/* pointer to the start of a path */
} PATH_T;

extern PATH_T to;
extern int Nflag, fflag, iflag, lflag, nflag, pflag, sflag, vflag;
#ifdef __APPLE__
extern int unix2003_compat;
extern int cflag;
extern int Sflag;
extern int Xflag;
#endif /* __APPLE__ */
extern volatile sig_atomic_t info;

__BEGIN_DECLS
int	copy_fifo(struct stat *, int);
int	copy_file(const FTSENT *, int);
int	copy_link(const FTSENT *, int);
int	copy_special(struct stat *, int);
int	setfile(struct stat *, int);
int	preserve_dir_acls(struct stat *, char *, char *);
int	preserve_fd_acls(int, int);
void	usage(void) __dead2;
__END_DECLS

#endif /* _CP_EXTERN_H_ */
