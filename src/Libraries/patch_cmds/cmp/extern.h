/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#define OK_EXIT		0
#define DIFF_EXIT	1
#define ERR_EXIT	2	/* error exit code */

int	c_link(const char *, off_t, const char *, off_t, off_t);
int	c_regular(int, const char *, off_t, off_t, int, const char *, off_t,
	    off_t, off_t);
int	c_special(int, const char *, off_t, int, const char *, off_t, off_t);
void	diffmsg(const char *, const char *, off_t, off_t, int, int);
void	eofmsg(const char *);

extern bool bflag, lflag, sflag, xflag, zflag;

#ifdef SIGINFO
extern volatile sig_atomic_t info;
#endif
