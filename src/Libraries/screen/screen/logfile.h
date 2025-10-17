/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
struct logfile
{
  struct logfile *next;
  FILE *fp;		/* a hopefully uniq filepointer to the log file */
  char *name;		/* the name. used to reopen, when stat fails. */
  int opencount;	/* synchronize logfopen() and logfclose() */
  int writecount;	/* increments at logfwrite(), counts write() and fflush() */
  int flushcount;	/* increments at logfflush(), zeroed at logfwrite() */
  struct stat *st;	/* how the file looks like */
};

/*
 * open a logfile, The second argument must be NULL, when the named file
 * is already a logfile or must be a appropriatly opened file pointer
 * otherwise.
 * example: l = logfopen(name, islogfile(name) : NULL ? fopen(name, "a"));
 */
struct logfile *logfopen __P((char *name, FILE *fp));

/*
 * lookup a logfile by name. This is useful, so that we can provide
 * logfopen with a nonzero second argument, exactly when needed. 
 * islogfile(NULL); returns nonzero if there are any open logfiles at all.
 */
int islogfile __P((char *name));

/* 
 * logfclose does free()
 */
int logfclose __P((struct logfile *));
int logfwrite __P((struct logfile *, char *, int));

/* 
 * logfflush should be called periodically. If no argument is passed,
 * all logfiles are flushed, else the specified file
 * the number of flushed filepointers is returned
 */
int logfflush __P((struct logfile *ifany));

/* 
 * a reopen function may be registered here, in case you want to bring your 
 * own (more secure open), it may come along with a private data pointer.
 * this function is called, whenever logfwrite/logfflush detect that the
 * file has been (re)moved, truncated or changed by someone else.
 * if you provide NULL as parameter to logreopen_register, the builtin
 * reopen function will be reactivated.
 */
void logreopen_register __P((int (*fn) __P((char *, int, struct logfile *)) ));

/* 
 * Your custom reopen function is required to reuse the exact
 * filedescriptor. 
 * See logfile.c for further specs and an example.
 *
 * lf_move_fd may help you here, if you do not have dup2(2).
 * It closes fd and opens wantfd to access whatever fd accessed.
 */
int lf_move_fd __P((int fd, int wantfd));
