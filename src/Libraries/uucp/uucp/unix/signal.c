/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include "uucp.h"

#include "uudefs.h"
#include "sysdep.h"
#include "system.h"

#include <errno.h>

/* Signal handling routines.  When we catch a signal, we want to set
   the appropriate elements of afSignal and afLog_signal to TRUE.  If
   we are on a system which restarts system calls, we may also want to
   longjmp out.  On a system which does not restart system calls,
   these signal handling routines are well-defined by ANSI C.  */

#if HAVE_RESTARTABLE_SYSCALLS
volatile sig_atomic_t fSjmp;
volatile jmp_buf sSjmp_buf;
#endif /* HAVE_RESTARTABLE_SYSCALLS */

/* Some systems, such as SunOS, have a SA_INTERRUPT bit that must be
   set in the sigaction structure to force system calls to be
   interrupted.  */
#ifndef SA_INTERRUPT
#define SA_INTERRUPT 0
#endif

/* The SVR3 sigset function can be called just like signal, unless
   system calls are restarted which is extremely unlikely; we prevent
   this case in sysh.unx.  */
#if HAVE_SIGSET && ! HAVE_SIGACTION && ! HAVE_SIGVEC
#define signal sigset
#endif

/* The sigvec structure changed from 4.2BSD to 4.3BSD.  These macros
   make the 4.3 code backward compatible.  */
#ifndef SV_INTERRUPT
#define SV_INTERRUPT 0
#endif
#if ! HAVE_SIGVEC_SV_FLAGS
#define sv_flags sv_onstack
#endif

/* Catch a signal.  Reinstall the signal handler if necessary, set the
   appropriate variables, and do a longjmp if necessary.  */

RETSIGTYPE
ussignal (isig)
     int isig;
{
  int iindex;

#if ! HAVE_SIGACTION && ! HAVE_SIGVEC && ! HAVE_SIGSET
  (void) signal (isig, ussignal);
#endif

  switch (isig)
    {
    default: iindex = INDEXSIG_SIGHUP; break;
#ifdef SIGINT
    case SIGINT: iindex = INDEXSIG_SIGINT; break;
#endif
#ifdef SIGQUIT
    case SIGQUIT: iindex = INDEXSIG_SIGQUIT; break;
#endif
#ifdef SIGTERM
    case SIGTERM: iindex = INDEXSIG_SIGTERM; break;
#endif
#ifdef SIGPIPE
    case SIGPIPE: iindex = INDEXSIG_SIGPIPE; break;
#endif
    }

  afSignal[iindex] = TRUE;
  afLog_signal[iindex] = TRUE;

#if HAVE_RESTARTABLE_SYSCALLS
  if (fSjmp)
    longjmp (sSjmp_buf, 1);
#endif /* HAVE_RESTARTABLE_SYSCALLS */
}

/* Prepare to catch a signal.  This is basically the ANSI C routine
   signal, but it uses sigaction or sigvec instead if they are
   available.  If fforce is FALSE, we do not set the signal if it is
   currently being ignored.  If pfignored is not NULL and fforce is
   FALSE, then *pfignored will be set to TRUE if the signal was
   previously being ignored (if fforce is TRUE the value returned in
   *pfignored is meaningless).  If we can't change the signal handler
   we give a fatal error.  */

void
usset_signal (isig, pfn, fforce, pfignored)
     int isig;
     RETSIGTYPE (*pfn) P((int));
     boolean fforce;
     boolean *pfignored;
{
#if HAVE_SIGACTION

  struct sigaction s;

  if (! fforce)
    {
      (void) (sigemptyset (&s.sa_mask));
      if (sigaction (isig, (struct sigaction *) NULL, &s) != 0)
	ulog (LOG_FATAL, "sigaction (%d): %s", isig, strerror (errno));

      if (s.sa_handler == SIG_IGN)
	{
	  if (pfignored != NULL)
	    *pfignored = TRUE;
	  return;
	}

      if (pfignored != NULL)
	*pfignored = FALSE;
    }

  s.sa_handler = pfn;
  (void) (sigemptyset (&s.sa_mask));
  s.sa_flags = SA_INTERRUPT;

  if (sigaction (isig, &s, (struct sigaction *) NULL) != 0)
    ulog (LOG_FATAL, "sigaction (%d): %s", isig, strerror (errno));

#else /* ! HAVE_SIGACTION */
#if HAVE_SIGVEC

  struct sigvec s;

  if (! fforce)
    {
      if (sigvec (isig, (struct sigvec *) NULL, &s) != 0)
	ulog (LOG_FATAL, "sigvec (%d): %s", isig, strerror (errno));

      if (s.sv_handler == SIG_IGN)
	{
	  if (pfignored != NULL)
	    *pfignored = TRUE;
	  return;
	}

      if (pfignored != NULL)
	*pfignored = FALSE;
    }

  s.sv_handler = pfn;
  s.sv_mask = 0;
  s.sv_flags = SV_INTERRUPT;

  if (sigvec (isig, &s, (struct sigvec *) NULL) != 0)
    ulog (LOG_FATAL, "sigvec (%d): %s", isig, strerror (errno));

#else /* ! HAVE_SIGVEC */

  if (! fforce)
    {
      if (signal (isig, SIG_IGN) == SIG_IGN)
	{
	  if (pfignored != NULL)
	    *pfignored = TRUE;
	  return;
	}

      if (pfignored != NULL)
	*pfignored = FALSE;
    }

  (void) signal (isig, pfn);

#endif /* ! HAVE_SIGVEC */
#endif /* ! HAVE_SIGACTION */
}

/* The routine called by the system independent code, which always
   uses the same signal handler.  */

void
usysdep_signal (isig)
     int isig;
{
  usset_signal (isig, ussignal, FALSE, (boolean *) NULL);
}
