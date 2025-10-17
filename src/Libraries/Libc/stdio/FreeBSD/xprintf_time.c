/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
#include <namespace.h>
#include <stdio.h>
#include <wchar.h>
#include <stdint.h>
#include <assert.h>
#include <sys/time.h>
#include "printf.h"
#include "xprintf_private.h"

__private_extern__ int
__printf_arginfo_time(const struct printf_info *pi, size_t n, int *argt)
{

	assert(n >= 1);
	argt[0] = PA_POINTER;
	return (1);
}
#define MINUTE 60
#define HOUR	(60 * MINUTE)
#define DAY	(24 * HOUR)
#define YEAR	(365 * DAY)

__private_extern__ int
__printf_render_time(struct __printf_io *io, const struct printf_info *pi, const void *const *arg)
{
	char buf[100];
	char *p;
	struct timeval *tv;
	struct timespec *ts;
	time_t *tp;
	intmax_t t, tx;
	int i, prec, nsec, ret;

	if (pi->is_long) {
		tv = *((struct timeval **)arg[0]);
		t = tv->tv_sec;
		nsec = tv->tv_usec * 1000;
		prec = 6;
	} else if (pi->is_long_double) {
		ts = *((struct timespec **)arg[0]);
		t = ts->tv_sec;
		nsec = ts->tv_nsec;
		prec = 9;
	} else {
		tp = *((time_t **)arg[0]);
		t = *tp;
		nsec = 0;
		prec = 0;
	}
	if (pi->is_long || pi->is_long_double) {
		if (pi->prec >= 0) 
			prec = pi->prec;
		if (prec == 0)
			nsec = 0;
	}

	p = buf;
	if (pi->alt) {
		tx = t;
		if (t >= YEAR) {
			p += sprintf(p, "%jdy", t / YEAR);
			t %= YEAR;
		}
		if (tx >= DAY && (t != 0 || prec != 0)) {
			p += sprintf(p, "%jdd", t / DAY);
			t %= DAY;
		}
		if (tx >= HOUR && (t != 0 || prec != 0)) {
			p += sprintf(p, "%jdh", t / HOUR);
			t %= HOUR;
		}
		if (tx >= MINUTE && (t != 0 || prec != 0)) {
			p += sprintf(p, "%jdm", t / MINUTE);
			t %= MINUTE;
		}
		if (t != 0 || tx == 0 || prec != 0)
			p += sprintf(p, "%jds", t);
	} else  {
		p += sprintf(p, "%jd", (intmax_t)t);
	}
	if (prec != 0) {
		for (i = prec; i < 9; i++)
			nsec /= 10;
		p += sprintf(p, ".%.*d", prec, nsec);
	}
	ret = __printf_out(io, pi, buf, p - buf);
	__printf_flush(io);
	return (ret);
}
