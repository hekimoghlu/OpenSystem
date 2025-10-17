/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#include <sys/queue.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <mach/policy.h>
#include <mach/task_info.h>
#include <mach/thread_info.h>

#include <TargetConditionals.h>

#if !defined(PS_ENTITLED) && TARGET_OS_OSX
#define	PS_ENTITLEMENT_ENFORCED	1
#else
#define	PS_ENTITLEMENT_ENFORCED 0
#endif /* !PS_ENTITLED && TARGET_OS_OSX */
#endif /* __APPLE__ */

#define	UNLIMITED	0	/* unlimited terminal width */
enum type { CHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, KPTR, PGTOK };

struct usave {
	struct	timeval u_start;
	struct	rusage u_ru;
	struct	rusage u_cru;
	char	u_acflag;
	char	u_valid;
};

#define KI_PROC(ki) (&(ki)->ki_p->kp_proc)
#define KI_EPROC(ki) (&(ki)->ki_p->kp_eproc)

typedef struct thread_values {
	struct thread_basic_info tb;
	/* struct policy_infos	schedinfo; */
	union {
		struct policy_timeshare_info tshare;
		struct policy_rr_info rr;
		struct policy_fifo_info fifo;
	} schedinfo;
} thread_values_t;

typedef struct kinfo {
	struct kinfo_proc *ki_p;	/* kinfo_proc structure */
	struct usave ki_u;	/* interesting parts of user */
	char *ki_args;		/* exec args */
	char *ki_env;		/* environment */
        task_port_t task;
	int state;
	int cpu_usage;
	int curpri;
	int basepri;
	int swapped;
	struct task_basic_info tasks_info;
	struct task_thread_times_info times;
	/* struct policy_infos	schedinfo; */
	union {
		struct policy_timeshare_info tshare;
		struct policy_rr_info rr;
		struct policy_fifo_info fifo;
	} schedinfo;
	int	invalid_tinfo;
        unsigned int	thread_count;
        thread_port_array_t thread_list;
        thread_values_t *thval;
	int	invalid_thinfo;
} KINFO;

/* Variables. */
typedef struct varent {
	STAILQ_ENTRY(varent) next_ve;
	const char *header;
	struct var *var;
} VARENT;

typedef struct var {
	const char *name;	/* name(s) of variable */
	const char *header;	/* default header */
	const char *alias;	/* aliases */
#define	COMM	0x01		/* needs exec arguments and environment (XXX) */
#define	LJUST	0x02		/* left adjust on output (trailing blanks) */
#define	USER	0x04		/* needs user structure */
#define	DSIZ	0x08		/* field size is dynamic*/
#define	INF127	0x10		/* values >127 displayed as 127 */
#ifdef __APPLE__
#define	ENTITLED	0x80000000	/* Needs entitlements */
#endif
	u_int	flag;
				/* output routine */
	void	(*oproc)(struct kinfo *, struct varent *);
				/* sizing routine*/
	int	(*sproc)(struct kinfo *);
	short	width;		/* printing width */
	/*
	 * The following (optional) elements are hooks for passing information
	 * to the generic output routine pvar (which prints simple elements
	 * from the well known kinfo_proc structure).
	 */
	size_t	off;		/* offset in structure */
	enum	type type;	/* type of element */
	const char *fmt;	/* printf format */
	short	dwidth;		/* dynamic printing width */
	/*
	 * glue to link selected fields together
	 */
} VAR;

#include "extern.h"
