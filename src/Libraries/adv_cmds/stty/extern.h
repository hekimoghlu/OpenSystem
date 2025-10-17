/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
int	c_cchars(const void *, const void *);
int	c_modes(const void *, const void *);
int	csearch(char ***, struct info *);
void	checkredirect(void);
void	gprint(struct termios *, struct winsize *, int);
void	gread(struct termios *, char *);
int	ksearch(char ***, struct info *);
int	msearch(char ***, struct info *);
void	optlist(void);
void	print(struct termios *, struct winsize *, int, enum FMT);
void	usage(void) __dead2;

extern struct cchar cchars1[], cchars2[];
