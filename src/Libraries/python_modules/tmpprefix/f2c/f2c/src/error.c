/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#include "defs.h"

 void
#ifdef KR_headers
warni(s, t)
	char *s;
	int t;
#else
warni(char *s, int t)
#endif
{
	char buf[100];
	sprintf(buf,s,t);
	warn(buf);
	}

 void
#ifdef KR_headers
warn1(s, t)
	char *s;
	char *t;
#else
warn1(const char *s, const char *t)
#endif
{
	char buff[100];
	sprintf(buff, s, t);
	warn(buff);
}

 void
#ifdef KR_headers
warn(s)
	char *s;
#else
warn(char *s)
#endif
{
	if(nowarnflag)
		return;
	if (infname && *infname)
		fprintf(diagfile, "Warning on line %ld of %s: %s\n",
			lineno, infname, s);
	else
		fprintf(diagfile, "Warning on line %ld: %s\n", lineno, s);
	fflush(diagfile);
	++nwarn;
}

 void
#ifdef KR_headers
errstr(s, t)
	char *s;
	char *t;
#else
errstr(const char *s, const char *t)
#endif
{
	char buff[100];
	sprintf(buff, s, t);
	err(buff);
}


 void
#ifdef KR_headers
erri(s, t)
	char *s;
	int t;
#else
erri(char *s, int t)
#endif
{
	char buff[100];
	sprintf(buff, s, t);
	err(buff);
}

 void
#ifdef KR_headers
errl(s, t)
	char *s;
	long t;
#else
errl(char *s, long t)
#endif
{
	char buff[100];
	sprintf(buff, s, t);
	err(buff);
}

 char *err_proc = 0;

 void
#ifdef KR_headers
err(s)
	char *s;
#else
err(char *s)
#endif
{
	if (err_proc)
		fprintf(diagfile,
			"Error processing %s before line %ld",
			err_proc, lineno);
	else
		fprintf(diagfile, "Error on line %ld", lineno);
	if (infname && *infname)
		fprintf(diagfile, " of %s", infname);
	fprintf(diagfile, ": %s\n", s);
	fflush(diagfile);
	++nerr;
}

 void
#ifdef KR_headers
yyerror(s)
	char *s;
#else
yyerror(char *s)
#endif
{
	err(s);
}


 void
#ifdef KR_headers
dclerr(s, v)
	char *s;
	Namep v;
#else
dclerr(const char *s, Namep v)
#endif
{
	char buff[100];

	if(v)
	{
		sprintf(buff, "Declaration error for %s: %s", v->fvarname, s);
		err(buff);
	}
	else
		errstr("Declaration error %s", s);
}


 void
#ifdef KR_headers
execerr(s, n)
	char *s;
	char *n;
#else
execerr(char *s, char *n)
#endif
{
	char buf1[100], buf2[100];

	sprintf(buf1, "Execution error %s", s);
	sprintf(buf2, buf1, n);
	err(buf2);
}


 void
#ifdef KR_headers
Fatal(t)
	char *t;
#else
Fatal(char *t)
#endif
{
	fprintf(diagfile, "Compiler error line %ld", lineno);
	if (infname)
		fprintf(diagfile, " of %s", infname);
	fprintf(diagfile, ": %s\n", t);
	done(3);
}



 void
#ifdef KR_headers
fatalstr(t, s)
	char *t;
	char *s;
#else
fatalstr(char *t, char *s)
#endif
{
	char buff[100];
	sprintf(buff, t, s);
	Fatal(buff);
}


 void
#ifdef KR_headers
fatali(t, d)
	char *t;
	int d;
#else
fatali(char *t, int d)
#endif
{
	char buff[100];
	sprintf(buff, t, d);
	Fatal(buff);
}


 void
#ifdef KR_headers
badthing(thing, r, t)
	char *thing;
	char *r;
	int t;
#else
badthing(char *thing, char *r, int t)
#endif
{
	char buff[50];
	sprintf(buff, "Impossible %s %d in routine %s", thing, t, r);
	Fatal(buff);
}


 void
#ifdef KR_headers
badop(r, t)
	char *r;
	int t;
#else
badop(char *r, int t)
#endif
{
	badthing("opcode", r, t);
}


 void
#ifdef KR_headers
badtag(r, t)
	char *r;
	int t;
#else
badtag(char *r, int t)
#endif
{
	badthing("tag", r, t);
}




 void
#ifdef KR_headers
badstg(r, t)
	char *r;
	int t;
#else
badstg(char *r, int t)
#endif
{
	badthing("storage class", r, t);
}



 void
#ifdef KR_headers
badtype(r, t)
	char *r;
	int t;
#else
badtype(char *r, int t)
#endif
{
	badthing("type", r, t);
}

 void
#ifdef KR_headers
many(s, c, n)
	char *s;
	char c;
	int n;
#else
many(char *s, char c, int n)
#endif
{
	char buff[250];

	sprintf(buff,
	    "Too many %s.\nTable limit now %d.\nTry rerunning with the -N%c%d option.\n",
	    s, n, c, 2*n);
	Fatal(buff);
}

 void
#ifdef KR_headers
err66(s)
	char *s;
#else
err66(char *s)
#endif
{
	errstr("Fortran 77 feature used: %s", s);
	--nerr;
}


 void
#ifdef KR_headers
errext(s)
	char *s;
#else
errext(char *s)
#endif
{
	errstr("f2c extension used: %s", s);
	--nerr;
}
