/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef MISC_H
# define MISC_H
# include <sys/types.h>
# include <stdarg.h>
# include <stdio.h>
# include "zconf.h"

# define min(a, b)	((a) < (b) ? (a) : (b))
# define max(a, b)	((a) > (b) ? (a) : (b))

extern	const	char	*getnameappendix (const char *progname, const char *basename);
extern	const	char	*getdefconfname (const char *view);
extern	int	fileexist (const char *name);
extern	size_t	filesize (const char *name);
extern	int	file_age (const char *fname);
extern	int	touch (const char *fname, time_t sec);
extern	int	linkfile (const char *fromfile, const char *tofile);
//extern	int	copyfile (const char *fromfile, const char *tofile);
extern	int	copyfile (const char *fromfile, const char *tofile, const char *dnskeyfile);
extern	int	copyzonefile (const char *fromfile, const char *tofile, const char *dnskeyfile);
extern	int	cmpfile (const char *file1, const char *file2);
extern	char	*str_delspace (char *s);
#if 1
extern	char	*domain_canonicdup (const char *s);
#else
extern	char	*str_tolowerdup (const char *s);
#endif
extern	int	in_strarr (const char *str, char *const arr[], int cnt);
extern	const	char	*splitpath (char *path, size_t  size, const char *filename);
extern	char	*pathname (char *name, size_t size, const char *path, const char *file, const char *ext);
extern	char	*time2str (time_t sec, int precision);
extern	char	*time2isostr (time_t sec, int precision);
extern	time_t	timestr2time (const char *timestr);
extern	int	is_keyfilename (const char *name);
extern	int	is_directory (const char *name);
extern	time_t	file_mtime (const char *fname);
extern	int	is_exec_ok (const char *prog);
extern	char	*age2str (time_t sec);
extern	time_t	stop_timer (time_t start);
extern	time_t	start_timer (void);
extern	void    error (char *fmt, ...);
extern	void    fatal (char *fmt, ...);
extern	void    logmesg (char *fmt, ...);
extern	void	verbmesg (int verblvl, const zconf_t *conf, char *fmt, ...);
extern	void	logflush (void);
extern	int	gensalt (char *salt, size_t saltsize, int saltbits, unsigned int seed);
extern	char	*str_untaint (char *str);
extern	char	*str_chop (char *str, char c);
extern	int	is_dotfilename (const char *name);
extern	void	parseurl (char *url, char **proto, char **host, char **port, char **para);
#endif
