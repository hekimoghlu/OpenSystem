/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
extern struct s_command *prog;
extern struct s_appends *appends;
extern regmatch_t *match;
extern size_t maxnsub;
extern u_long linenum;
extern unsigned int appendnum;
extern int aflag, eflag, nflag;
extern const char *fname, *outfname;
extern FILE *infile, *outfile;
extern int rflags;	/* regex flags to use */
extern const char *inplace;
extern int quit;

void	 cfclose(struct s_command *, struct s_command *);
void	 compile(void);
void	 cspace(SPACE *, const char *, size_t, enum e_spflag);
char	*cu_fgets(char *, int, int *);
int	 mf_fgets(SPACE *, enum e_spflag);
int	 lastline(void);
void	 process(void);
void	 resetstate(void);
char	*strregerror(int, regex_t *);
