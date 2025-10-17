/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include <sys/param.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "extern.h"

static u_char *remmap(u_char *, int, off_t);
static void segv_handler(int);
#define MMAP_CHUNK (8*1024*1024)

#define ROUNDPAGE(i) ((i) & ~pagemask)

int
c_regular(int fd1, const char *file1, off_t skip1, off_t len1,
    int fd2, const char *file2, off_t skip2, off_t len2, off_t limit)
{
	struct sigaction act, oact;
#ifndef __APPLE__
	cap_rights_t rights;
#endif
	u_char ch, *p1, *p2, *m1, *m2, *e1, *e2;
	off_t byte, length, line;
	off_t pagemask, off1, off2;
	size_t pagesize;
	int dfound;

	if (skip1 > len1) {
		eofmsg(file1);
		return (DIFF_EXIT);
	}
	len1 -= skip1;
	if (skip2 > len2) {
		eofmsg(file2);
		return (DIFF_EXIT);
	}
	len2 -= skip2;

	if (sflag && len1 != len2)
		return (DIFF_EXIT);

	pagesize = getpagesize();
	pagemask = (off_t)pagesize - 1;
	off1 = ROUNDPAGE(skip1);
	off2 = ROUNDPAGE(skip2);

	length = MIN(len1, len2);
	if (limit > 0)
		length = MIN(length, limit);

	if ((m1 = remmap(NULL, fd1, off1)) == NULL) {
		return (c_special(fd1, file1, skip1, fd2, file2, skip2, limit));
	}

	if ((m2 = remmap(NULL, fd2, off2)) == NULL) {
		munmap(m1, MMAP_CHUNK);
		return (c_special(fd1, file1, skip1, fd2, file2, skip2, limit));
	}

#ifndef __APPLE__
	if (caph_rights_limit(fd1, cap_rights_init(&rights, CAP_MMAP_R)) < 0)
		err(1, "unable to limit rights for %s", file1);
	if (caph_rights_limit(fd2, cap_rights_init(&rights, CAP_MMAP_R)) < 0)
		err(1, "unable to limit rights for %s", file2);
	if (caph_enter() < 0)
		err(ERR_EXIT, "unable to enter capability mode");
#endif

	sigemptyset(&act.sa_mask);
	act.sa_flags = SA_NODEFER;
	act.sa_handler = segv_handler;
	if (sigaction(SIGSEGV, &act, &oact))
		err(ERR_EXIT, "sigaction()");

	dfound = 0;
	e1 = m1 + MMAP_CHUNK;
	e2 = m2 + MMAP_CHUNK;
	p1 = m1 + (skip1 - off1);
	p2 = m2 + (skip2 - off2);

	for (byte = line = 1; length--; ++byte) {
#ifdef SIGINFO
		if (info) {
			(void)fprintf(stderr, "%s %s char %zu line %zu\n",
			    file1, file2, (size_t)byte, (size_t)line);
			info = 0;
		}
#endif
		if ((ch = *p1) != *p2) {
			dfound = 1;
			if (xflag) {
				(void)printf("%08llx %02x %02x\n",
				    (long long)byte - 1, ch, *p2);
			} else if (lflag) {
				if (bflag)
					(void)printf("%6lld %3o %c %3o %c\n",
					    (long long)byte, ch, ch, *p2, *p2);
				else
					(void)printf("%6lld %3o %3o\n",
					    (long long)byte, ch, *p2);
			} else {
				diffmsg(file1, file2, byte, line, ch, *p2);
				return (DIFF_EXIT);
			}
		}
		if (ch == '\n')
			++line;
		if (++p1 == e1) {
			off1 += MMAP_CHUNK;
			if ((p1 = m1 = remmap(m1, fd1, off1)) == NULL) {
				munmap(m2, MMAP_CHUNK);
				err(ERR_EXIT, "remmap %s", file1);
			}
			e1 = m1 + MMAP_CHUNK;
		}
		if (++p2 == e2) {
			off2 += MMAP_CHUNK;
			if ((p2 = m2 = remmap(m2, fd2, off2)) == NULL) {
				munmap(m1, MMAP_CHUNK);
				err(ERR_EXIT, "remmap %s", file2);
			}
			e2 = m2 + MMAP_CHUNK;
		}
	}
	munmap(m1, MMAP_CHUNK);
	munmap(m2, MMAP_CHUNK);

	if (sigaction(SIGSEGV, &oact, NULL))
		err(ERR_EXIT, "sigaction()");

	if (len1 != len2) {
		eofmsg(len1 > len2 ? file2 : file1);
		return (DIFF_EXIT);
	}
	return (dfound ? DIFF_EXIT : 0);
}

static u_char *
remmap(u_char *mem, int fd, off_t offset)
{
#if !TARGET_OS_WATCH
	if (mem != NULL)
		munmap(mem, MMAP_CHUNK);
	mem = mmap(NULL, MMAP_CHUNK, PROT_READ, MAP_SHARED, fd, offset);
	if (mem == MAP_FAILED)
		return (NULL);
	madvise(mem, MMAP_CHUNK, MADV_SEQUENTIAL);
#endif /* !TARGET_OS_WATCH */
	return (mem);
}

static void
segv_handler(int sig __unused) {
	static const char msg[] = "cmp: Input/output error (caught SIGSEGV)\n";

	write(STDERR_FILENO, msg, sizeof(msg));
	_exit(EXIT_FAILURE);
}
