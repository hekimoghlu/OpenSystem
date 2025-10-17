/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
struct shparam {
	int nparam;		/* # of positional parameters (without $0) */
	unsigned char malloc;	/* if parameter list dynamically allocated */
	unsigned char reset;	/* if getopts has been reset */
	char **p;		/* parameter list */
	char **optp;		/* parameter list for getopts */
	char **optnext;		/* next parameter to be processed by getopts */
	char *optptr;		/* used by getopts */
};



#define eflag optval[0]
#define fflag optval[1]
#define Iflag optval[2]
#define iflag optval[3]
#define mflag optval[4]
#define nflag optval[5]
#define sflag optval[6]
#define xflag optval[7]
#define vflag optval[8]
#define Vflag optval[9]
#define	Eflag optval[10]
#define	Cflag optval[11]
#define	aflag optval[12]
#define	bflag optval[13]
#define	uflag optval[14]
#define	privileged optval[15]
#define	Tflag optval[16]
#define	Pflag optval[17]
#define	hflag optval[18]
#define	nologflag optval[19]

#define NSHORTOPTS	19
#define NOPTS		20

extern char optval[NOPTS];
extern const char optletter[NSHORTOPTS];
#ifdef DEFINE_OPTIONS
char optval[NOPTS];
const char optletter[NSHORTOPTS] = "efIimnsxvVECabupTPh";
static const unsigned char optname[] =
	"\007errexit"
	"\006noglob"
	"\011ignoreeof"
	"\013interactive"
	"\007monitor"
	"\006noexec"
	"\005stdin"
	"\006xtrace"
	"\007verbose"
	"\002vi"
	"\005emacs"
	"\011noclobber"
	"\011allexport"
	"\006notify"
	"\007nounset"
	"\012privileged"
	"\012trapsasync"
	"\010physical"
	"\010trackall"
	"\005nolog"
;
#endif


extern char *minusc;		/* argument to -c option */
extern char *arg0;		/* $0 */
extern struct shparam shellparam;  /* $@ */
extern char **argptr;		/* argument list for builtin commands */
extern char *shoptarg;		/* set by nextopt */
extern char *nextopt_optptr;	/* used by nextopt */

void procargs(int, char **);
void optschanged(void);
void freeparam(struct shparam *);
int nextopt(const char *);
void getoptsreset(const char *);
