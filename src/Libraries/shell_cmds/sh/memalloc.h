/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include <string.h>

struct stackmark {
	struct stack_block *stackp;
	char *stacknxt;
	int stacknleft;
};


extern char *stacknxt;
extern int stacknleft;
extern char *sstrend;

pointer ckmalloc(size_t);
pointer ckrealloc(pointer, int);
void ckfree(pointer);
char *savestr(const char *);
pointer stalloc(int);
void stunalloc(pointer);
char *stsavestr(const char *);
void setstackmark(struct stackmark *);
void popstackmark(struct stackmark *);
char *growstackstr(void);
char *makestrspace(int, char *);
char *stputbin(const char *data, size_t len, char *p);
char *stputs(const char *data, char *p);



#define stackblock() stacknxt
#define stackblocksize() stacknleft
#define grabstackblock(n) stalloc(n)
#define STARTSTACKSTR(p)	p = stackblock()
#define STPUTC(c, p)	do { if (p == sstrend) p = growstackstr(); *p++ = (c); } while(0)
#define CHECKSTRSPACE(n, p)	{ if ((size_t)(sstrend - p) < n) p = makestrspace(n, p); }
#define USTPUTC(c, p)	(*p++ = (c))
/*
 * STACKSTRNUL's use is where we want to be able to turn a stack
 * (non-sentinel, character counting string) into a C string,
 * and later pretend the NUL is not there.
 * Note: Because of STACKSTRNUL's semantics, STACKSTRNUL cannot be used
 * on a stack that will grabstackstr()ed.
 */
#define STACKSTRNUL(p)	(p == sstrend ? (p = growstackstr(), *p = '\0') : (*p = '\0'))
#define STUNPUTC(p)	(--p)
#define STTOPC(p)	p[-1]
#define STADJUST(amount, p)	(p += (amount))
#define grabstackstr(p)	stalloc((char *)p - stackblock())
#define ungrabstackstr(s, p)	stunalloc((s))
#define STPUTBIN(s, len, p)	p = stputbin((s), (len), p)
#define STPUTS(s, p)	p = stputs((s), p)
