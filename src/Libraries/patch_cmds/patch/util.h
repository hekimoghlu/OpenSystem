/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifdef __APPLE__
char		*fetchname(const char *, bool *, int, const char **);
#else
char		*fetchname(const char *, bool *, int);
#endif
int		backup_file(const char *);
#ifdef __APPLE__
int		move_file(const char *, const char *, bool);
#else
int		move_file(const char *, const char *);
#endif
int		copy_file(const char *, const char *);
void		say(const char *, ...)
		    __attribute__((__format__(__printf__, 1, 2)));
void		fatal(const char *, ...)
		    __attribute__((__format__(__printf__, 1, 2)));
void		pfatal(const char *, ...)
		    __attribute__((__format__(__printf__, 1, 2)));
void		ask(const char *, ...)
		    __attribute__((__format__(__printf__, 1, 2)));
char		*savestr(const char *);
#ifdef __APPLE__
char		*saveline(const char *, size_t, size_t *);
#endif
char		*xstrdup(const char *);
void		set_signals(int);
void		ignore_signals(void);
void		makedirs(const char *, bool);
void		version(void);
void		my_exit(int) __attribute__((noreturn));

#ifdef __APPLE__
const char	*quoted_name(const char *filename);
#endif

/* in mkpath.c */
extern int mkpath(char *);
