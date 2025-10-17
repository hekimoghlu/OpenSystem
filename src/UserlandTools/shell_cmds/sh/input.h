/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
/* PEOF (the end of file marker) is defined in syntax.h */

/*
 * The input line number.  Input.c just defines this variable, and saves
 * and restores it when files are pushed and popped.  The user of this
 * package must set its value.
 */
extern int plinno;
extern int parsenleft;		/* number of characters left in input buffer */
extern const char *parsenextc;	/* next character in input buffer */

struct alias;
struct parsefile;

void resetinput(void);
int pgetc(void);
int preadbuffer(void);
int preadateof(void);
void pungetc(void);
void pushstring(const char *, int, struct alias *);
void setinputfile(const char *, int);
void setinputfd(int, int);
void setinputstring(const char *, int);
void popfile(void);
struct parsefile *getcurrentfile(void);
void popfilesupto(struct parsefile *);
void popallfiles(void);
void closescript(void);

#define pgetc_macro()	(--parsenleft >= 0? *parsenextc++ : preadbuffer())
