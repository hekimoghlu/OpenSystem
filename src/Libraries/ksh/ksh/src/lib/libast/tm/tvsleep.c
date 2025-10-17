/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#pragma prototyped

#include <tv.h>
#include <tm.h>

#include "FEATURE/tvlib"

#if !_lib_nanosleep
# if _lib_select
#  if _sys_select
#   include <sys/select.h>
#  else
#   include <sys/socket.h>
#  endif
# else
#  if !_lib_usleep
#   if _lib_poll_notimer
#    undef _lib_poll
#   endif
#   if _lib_poll
#    include <poll.h>
#   endif
#  endif
# endif
#endif

/*
 * sleep for tv
 * non-zero exit if sleep did not complete
 * with remaining time in rv
 */

int
tvsleep(register const Tv_t* tv, register Tv_t* rv)
{

#if _lib_nanosleep

	struct timespec	stv;
	struct timespec	srv;
	int		r;

	stv.tv_sec = tv->tv_sec;
	stv.tv_nsec = tv->tv_nsec;
	if ((r = nanosleep(&stv, &srv)) && rv)
	{
		rv->tv_sec = srv.tv_sec;
		rv->tv_nsec = srv.tv_nsec;
	}
	return r;

#else

#if _lib_select

	struct timeval	stv;

	stv.tv_sec = tv->tv_sec;
	stv.tv_usec = tv->tv_nsec / 1000;
	if (select(0, NiL, NiL, NiL, &stv) < 0)
	{
		if (rv)
			*rv = *tv;
		return -1;
	}
	if (rv)
	{
		rv->tv_sec = stv.tv_sec;
		rv->tv_nsec = stv.tv_usec * 1000;
	}
	return 0;

#else

	unsigned int		s = tv->tv_sec;
	uint32_t		n = tv->tv_nsec;

#if _lib_usleep


	unsigned long		t;

	if (t = (n + 999L) / 1000L)
	{
		usleep(t);
		s -= t / 1000000L;
		n = 0;
	}

#else

#if _lib_poll

	struct pollfd		pfd;
	int			t;

	if ((t = (n + 999999L) / 1000000L) > 0)
	{
		poll(&pfd, 0, t);
		s -= t / 1000L;
		n = 0;
	}

#endif

#endif

	if ((s += (n + 999999999L) / 1000000000L) && (s = sleep(s)))
	{
		if (rv)
		{
			rv->tv_sec = s;
			rv->tv_nsec = 0;
		}
		return -1;
	}
	return 0;

#endif

#endif

}
