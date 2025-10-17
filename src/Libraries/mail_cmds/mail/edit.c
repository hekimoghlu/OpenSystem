/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)edit.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "rcv.h"
#include <fcntl.h>
#include "extern.h"

/*
 * Mail -- a mail program
 *
 * Perform message editing functions.
 */

/*
 * Edit a message list.
 */
int
editor(int *msgvec)
{

	return (edit1(msgvec, 'e'));
}

/*
 * Invoke the visual editor on a message list.
 */
int
visual(int *msgvec)
{

	return (edit1(msgvec, 'v'));
}

/*
 * Edit a message by writing the message into a funnily-named file
 * (which should not exist) and forking an editor on it.
 * We get the editor from the stuff above.
 */
int
edit1(int *msgvec, int type)
{
	int c, i;
	FILE *fp;
	struct message *mp;
	off_t size;

	/*
	 * Deal with each message to be edited . . .
	 */
	for (i = 0; i < msgCount && msgvec[i]; i++) {
		sig_t sigint;

		if (i > 0) {
			char buf[100];
			char *p;

			printf("Edit message %d [ynq]? ", msgvec[i]);
			if (fgets(buf, sizeof(buf), stdin) == NULL)
				break;
			for (p = buf; *p == ' ' || *p == '\t'; p++)
				;
			if (*p == 'q')
				break;
			if (*p == 'n')
				continue;
		}
		dot = mp = &message[msgvec[i] - 1];
		touch(mp);
		sigint = signal(SIGINT, SIG_IGN);
		fp = run_editor(setinput(mp), mp->m_size, type, readonly);
		if (fp != NULL) {
			(void)fseeko(otf, (off_t)0, SEEK_END);
			size = ftello(otf);
			mp->m_block = blockof(size);
			mp->m_offset = boffsetof(size);
			mp->m_size = (long)fsize(fp);
			mp->m_lines = 0;
			mp->m_flag |= MODIFY;
			rewind(fp);
			while ((c = getc(fp)) != EOF) {
				if (c == '\n')
					mp->m_lines++;
				if (putc(c, otf) == EOF)
					break;
			}
			if (ferror(otf))
				warnx("/tmp");
			(void)Fclose(fp);
		}
		(void)signal(SIGINT, sigint);
	}
	return (0);
}

/*
 * Run an editor on the file at "fpp" of "size" bytes,
 * and return a new file pointer.
 * Signals must be handled by the caller.
 * "Type" is 'e' for _PATH_EX, 'v' for _PATH_VI.
 */
FILE *
run_editor(FILE *fp, off_t size, int type, int readonly)
{
	FILE *nf = NULL;
	int t;
	time_t modtime;
	char *edit, tempname[PATHSIZE];
	struct stat statb;

	(void)snprintf(tempname, sizeof(tempname),
	    "%s/mail.ReXXXXXXXXXX", tmpdir);
	if ((t = mkstemp(tempname)) == -1 ||
	    (nf = Fdopen(t, "w")) == NULL) {
		warn("%s", tempname);
		goto out;
	}
	if (readonly && fchmod(t, 0400) == -1) {
		warn("%s", tempname);
		(void)rm(tempname);
		goto out;
	}
	if (size >= 0)
		while (--size >= 0 && (t = getc(fp)) != EOF)
			(void)putc(t, nf);
	else
		while ((t = getc(fp)) != EOF)
			(void)putc(t, nf);
	(void)fflush(nf);
	if (fstat(fileno(nf), &statb) < 0)
		modtime = 0;
	else
		modtime = statb.st_mtime;
	if (ferror(nf)) {
		(void)Fclose(nf);
		warnx("%s", tempname);
		(void)rm(tempname);
		nf = NULL;
		goto out;
	}
	if (Fclose(nf) < 0) {
		warn("%s", tempname);
		(void)rm(tempname);
		nf = NULL;
		goto out;
	}
	nf = NULL;
	if (((edit = value(type == 'e' ? "EDITOR" : "VISUAL")) == NULL) || *edit == '\0')
		edit = type == 'e' ? _PATH_EX : _PATH_VI;

	if (run_command(edit, 0, -1, -1, tempname, NULL) < 0) {
		(void)rm(tempname);
		goto out;
	}
	/*
	 * If in read only mode or file unchanged, just remove the editor
	 * temporary and return.
	 */
	if (readonly) {
		(void)rm(tempname);
		goto out;
	}
	if (stat(tempname, &statb) < 0) {
		warn("%s", tempname);
		goto out;
	}
	if (modtime == statb.st_mtime) {
		(void)rm(tempname);
		goto out;
	}
	/*
	 * Now switch to new file.
	 */
	if ((nf = Fopen(tempname, "a+")) == NULL) {
		warn("%s", tempname);
		(void)rm(tempname);
		goto out;
	}
	(void)rm(tempname);
out:
	return (nf);
}
