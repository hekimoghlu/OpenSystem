/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#ifndef	_EXTERN_H_
#define	_EXTERN_H_

extern char tbuf[1024];			/* Temp buffer for anybody. */
extern int entries;			/* Number of people. */
extern DB *db;				/* Database. */
extern int d_first;
extern sa_family_t family;
extern int gflag;
extern int lflag;
extern time_t now;
extern int oflag;
extern int pplan;			/* don't show .plan/.project */
extern int invoker_root;		/* Invoked by root */

void	 enter_lastlog(PERSON *);
PERSON	*enter_person(struct passwd *);
void	 enter_where(struct utmpx *, PERSON *);
PERSON	*find_person(char *);
int	 hide(struct passwd *);
void	 lflag_print(void);
int	 match(struct passwd *, const char *);
void	 netfinger(char *);
PERSON	*palloc(void);
char	*prphone(char *);
void	 sflag_print(void);
int	 show_text(const char *, const char *, const char *);

#endif /* !_EXTERN_H_ */
