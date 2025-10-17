/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef LOG_H
# define LOG_H
# include <sys/types.h>
# include <stdarg.h>
# include <stdio.h>
# include <time.h>
# include <syslog.h>

#ifndef LOG_FNAMETMPL
# define	LOG_FNAMETMPL	"/zkt-%04d-%02d-%02dT%02d%02d%02dZ+log"
#endif

#ifndef LOG_DOMAINTMPL
# define	LOG_DOMAINTMPL	"zktlog-%s"
#endif


typedef enum {
	LG_NONE = 0,
	LG_DEBUG,
	LG_INFO,
	LG_NOTICE,
	LG_WARNING,
	LG_ERROR,
	LG_FATAL
} lg_lvl_t;

extern	lg_lvl_t	lg_str2lvl (const char *name);
extern	int	lg_str2syslog (const char *facility);
extern	const	char	*lg_lvl2str (lg_lvl_t level);
extern	lg_lvl_t	lg_lvl2syslog (lg_lvl_t level);
extern	long	lg_geterrcnt (void);
extern	long	lg_seterrcnt (long value);
extern	long	lg_reseterrcnt (void);
extern	int	lg_open (const char *progname, const char *facility, const char *syslevel, const char *path, const char *file, const char *filelevel);
extern	int	lg_close (void);
extern	int	lg_zone_start (const char *dir, const char *domain);
extern	int	lg_zone_end (void);
extern	void	lg_args (lg_lvl_t level, int argc, char * const argv[]);
extern	void	lg_mesg (int level, char *fmt, ...);
#endif
