/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <sys/file.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <utmp.h>

static int logout P((const char *zdev));
static void logwtmp P((const char *zdev, const char *zname,
		      const char *zhost));

int
main (argc, argv)
     int argc;
     char **argv;
{
  char *z;

  if (argc != 2
      || strncmp (argv[1], "/dev/", sizeof "/dev/" - 1) != 0)
    {
      fprintf (stderr, "Usage: tstout device\n");
      exit (EXIT_FAILURE);
    }

  z = argv[1] + 5;

  if (logout (z))
    logwtmp (z, "", "");

  chmod (argv[1], 0666);
  chown (argv[1], 0, 0);

  *z = 'p';
  chmod (argv[1], 0666);
  chown (argv[1], 0, 0);

  exit (EXIT_SUCCESS);
}

/*
 * Copyright (c) 1988 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, Berkeley.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)logout.c	5.2 (Berkeley) 2/17/89";
#endif /* LIBC_SCCS and not lint */

#define	UTMPFILE	"/etc/utmp"

/* 0 on failure, 1 on success */

static int
logout(line)
	register const char *line;
{
	register FILE *fp;
	struct utmp ut;
	int rval;
	time_t time();

	if (!(fp = fopen(UTMPFILE, "r+")))
		return(0);
	rval = 0;
	while (fread((char *)&ut, sizeof(struct utmp), 1, fp) == 1) {
		if (!ut.ut_name[0] ||
		    strncmp(ut.ut_line, line, sizeof(ut.ut_line)))
			continue;
		bzero(ut.ut_name, sizeof(ut.ut_name));
		bzero(ut.ut_host, sizeof(ut.ut_host));
		(void)time((time_t *)&ut.ut_time);
		(void)fseek(fp, (long)-sizeof(struct utmp), L_INCR);
		(void)fwrite((char *)&ut, sizeof(struct utmp), 1, fp);
		(void)fseek(fp, (long)0, L_INCR);
		rval = 1;
	}
	(void)fclose(fp);
	return(rval);
}

/*
 * Copyright (c) 1988 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, Berkeley.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)logwtmp.c	5.2 (Berkeley) 9/20/88";
#endif /* LIBC_SCCS and not lint */

#define	WTMPFILE	"/usr/adm/wtmp"

static void
logwtmp(line, name, host)
     const char *line, *name, *host;
{
	struct utmp ut;
	struct stat buf;
	int fd;
	time_t time();
	char *strncpy();

	if ((fd = open(WTMPFILE, O_WRONLY|O_APPEND, 0)) < 0)
		return;
	if (!fstat(fd, &buf)) {
		(void)strncpy(ut.ut_line, line, sizeof(ut.ut_line));
		(void)strncpy(ut.ut_name, name, sizeof(ut.ut_name));
		(void)strncpy(ut.ut_host, host, sizeof(ut.ut_host));
		(void)time((time_t *)&ut.ut_time);
		if (write(fd, (char *)&ut, sizeof(struct utmp)) !=
		    sizeof(struct utmp))
			(void)ftruncate(fd, buf.st_size);
	}
	(void)close(fd);
}
