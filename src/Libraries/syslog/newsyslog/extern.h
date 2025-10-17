/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#include <sys/cdefs.h>
#include <time.h>

#define	PTM_PARSE_ISO8601	0x0001	/* Parse ISO-standard format */
#define	PTM_PARSE_DWM		0x0002	/* Parse Day-Week-Month format */
#define	PTM_PARSE_MATCHDOM	0x0004	/* If the user specifies a day-of-month,
					 * then the result should be a month
					 * which actually has that day.  Eg:
					 * the user requests "day 31" when
					 * the present month is February. */

struct ptime_data;

/* Some global variables from newsyslog.c which might be of interest */
extern int	 dbg_at_times;		/* cmdline debugging option */
extern int	 noaction;		/* command-line option */
extern int	 verbose;		/* command-line option */
extern struct ptime_data *dbg_timenow;

__BEGIN_DECLS
struct ptime_data *ptime_init(const struct ptime_data *_optsrc);
int		 ptime_adjust4dst(struct ptime_data *_ptime, const struct
		    ptime_data *_dstsrc);
int		 ptime_free(struct ptime_data *_ptime);
int		 ptime_relparse(struct ptime_data *_ptime, int _parseopts,
		    time_t _basetime, const char *_str);
const char	*ptimeget_ctime(const struct ptime_data *_ptime);
double		 ptimeget_diff(const struct ptime_data *_minuend,
		    const struct ptime_data *_subtrahend);
time_t		 ptimeget_secs(const struct ptime_data *_ptime);
int		 ptimeset_nxtime(struct ptime_data *_ptime);
int		 ptimeset_time(struct ptime_data *_ptime, time_t _secs);
__END_DECLS
