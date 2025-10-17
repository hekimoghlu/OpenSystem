/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#ifndef _zzzzzz_RCFILE_H_
#define _zzzzzz_RCFILE_H_

/*
 * A unified options parser
 */
enum opt_argtype {OPTARG_STR, OPTARG_INT, OPTARG_BOOL};

#define	OPTFL_NONE	0x0000
#define	OPTFL_HAVEMIN	0x0001
#define	OPTFL_HAVEMAX	0x0002
#define	OPTFL_MINMAX	NAFL_HAVEMIN | NAFL_HAVEMAX

struct opt_args;

typedef int opt_callback_t (struct opt_args*);

struct opt_args {
	enum opt_argtype type;
	int	opt;		/* command line option */
	char *	name;		/* rc file equiv */
	int	flag;		/* OPTFL_* */
	int	ival;		/* int/bool values, or max len for str value */
	char *	str;		/* string value */
	int	min;		/* min for ival */
	int	max;		/* max for ival */
	opt_callback_t *fn;	/* call back to validate */
};


extern int cf_opterr, cf_optind, cf_optopt, cf_optreset;
extern const char *cf_optarg;

__BEGIN_DECLS

struct rcfile;

int  rc_close(struct rcfile *);
int  rc_getstringptr(struct rcfile *, const char *, const char *, char **);
int  rc_getstring(struct rcfile *, const char *, const char *, size_t, char *);
int  rc_getint(struct rcfile *, const char *, const char *, int *);
int  rc_getbool(struct rcfile *, const char *, const char *, int *);
struct rcfile * smb_open_rcfile(int noUserPrefs);;

__END_DECLS

#endif	/* _zzzzzz_RCFILE_H_ */
