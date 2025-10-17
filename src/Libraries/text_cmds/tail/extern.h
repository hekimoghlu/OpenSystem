/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#define	WR(p, size) do { \
	ssize_t res; \
	res = write(STDOUT_FILENO, p, size); \
	if (res != (ssize_t)size) { \
		if (res == -1) \
			oerr(); \
		else \
			errx(1, "stdout"); \
	} \
} while (0)

#define TAILMAPLEN (4<<20)

struct mapinfo {
	off_t	mapoff;
	off_t	maxoff;
	size_t	maplen;
	char	*start;
	int	fd;
};

struct file_info {
	FILE *fp;
	const char *file_name;
	struct stat st;
};

typedef struct file_info file_info_t;

enum STYLE { NOTSET = 0, FBYTES, FLINES, RBYTES, RLINES, REVERSE };

void follow(file_info_t *, enum STYLE, off_t);
void forward(FILE *, const char *, enum STYLE, off_t, struct stat *);
void reverse(FILE *, const char *, enum STYLE, off_t, struct stat *);

int bytes(FILE *, const char *, off_t);
int lines(FILE *, const char *, off_t);

void ierr(const char *);
void oerr(void);
int mapprint(struct mapinfo *, off_t, off_t);
int maparound(struct mapinfo *, off_t);
void printfn(const char *, int);

extern int Fflag, fflag, qflag, rflag, rval, no_files, vflag;
#ifndef __APPLE__
extern fileargs_t *fa;
#endif
