/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
struct arglist {
	char **args;
	int count;
	int capacity;
	char *smallarg[1];
};

/*
 * expandarg() flags
 */
#define EXP_SPLIT	0x1	/* perform word splitting */
#define EXP_TILDE	0x2	/* do normal tilde expansion */
#define	EXP_VARTILDE	0x4	/* expand tildes in an assignment */
#define EXP_CASE	0x10	/* keeps quotes around for CASE pattern */
#define EXP_SPLIT_LIT	0x20	/* IFS split literal text ${v+-a b c} */
#define EXP_LIT_QUOTED	0x40	/* for EXP_SPLIT_LIT, start off quoted */
#define EXP_GLOB	0x80	/* perform file globbing */

#define EXP_FULL	(EXP_SPLIT | EXP_GLOB)


void emptyarglist(struct arglist *);
void appendarglist(struct arglist *, char *);
union node;
void expandarg(union node *, struct arglist *, int);
void rmescapes(char *);
int casematch(union node *, const char *);
