/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
extern int eopterr;
extern int eoptind;
extern int eoptopt;
extern char *eoptarg;

void	 addnum(char *, int, int);
int	 egetopt(int, char * const *, const char *);
void	 flsh_errs(void);
int	 horzcol(int, char **);
int	 inln(FILE *, char *, int, int *, int, int *);
int	 inskip(FILE *, int, int);
void	 mfail(void);
int	 mulfile(int, char **);
FILE	*nxtfile(int, char **, const char **, char *, int);
int	 onecol(int, char **);
int	 otln(char *, int, int *, int *, int);
void	 pfail(void);
int	 prhead(char *, const char *, int);
int	 prtail(int, int);
int	 setup(int, char **);
void	 terminate(int);
void	 usage(void);
int	 vertcol(int, char **);
