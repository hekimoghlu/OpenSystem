/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#ifndef _EXTERN_H_
#define _EXTERN_H_

#include "mtree.h"

extern uint32_t crc_total;

#ifdef _FTS_H_
int	 compare(char *, NODE *, FTSENT *);
#endif
int	 crc(int, uint32_t *, off_t *);
void	 cwalk(void);
char	*flags_to_string(u_long);
char	*escape_path(char *string);
struct timespec	ptime(char *path, int *supported);

const char	*inotype(u_int);
u_int64_t	 parsekey(char *, int *);
char	*rlink(char *);
NODE	*mtree_readspec(FILE *fi);
int	mtree_verifyspec(FILE *fi);
int	mtree_specspec(FILE *fi, FILE *fj);

int	 check_excludes(const char *, const char *);
void	 init_excludes(void);
void	 read_excludes_file(const char *);
const char * ftype(u_int type);

extern int ftsoptions;
extern int xattr_options;
extern u_int64_t keys;
extern int lineno;
extern int dflag, eflag, iflag, nflag, qflag, rflag, sflag, uflag, wflag, mflag, tflag, xflag;
extern int insert_mod, insert_birth, insert_access, insert_change, insert_parent;
extern struct timespec ts;
#ifdef MAXPATHLEN
extern char fullpath[MAXPATHLEN];
#endif

#endif /* _EXTERN_H_ */
