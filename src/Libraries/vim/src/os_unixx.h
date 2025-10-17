/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
/*
 * os_unixx.h -- include files that are only used in os_unix.c
 */

// Sun's sys/ioctl.h redefines symbols from termio world
#if defined(HAVE_SYS_IOCTL_H) && !defined(SUN_SYSTEM)
# include <sys/ioctl.h>
#endif

#ifndef USE_SYSTEM	// use fork/exec to start the shell

# if defined(HAVE_SYS_WAIT_H) || defined(HAVE_UNION_WAIT)
#  include <sys/wait.h>
# endif

# ifndef WEXITSTATUS
#  ifdef HAVE_UNION_WAIT
#   define WEXITSTATUS(stat_val) ((stat_val).w_T.w_Retcode)
#  else
#   define WEXITSTATUS(stat_val) (((stat_val) >> 8) & 0377)
#  endif
# endif

# ifndef WIFEXITED
#  ifdef HAVE_UNION_WAIT
#   define WIFEXITED(stat_val) ((stat_val).w_T.w_Termsig == 0)
#  else
#   define WIFEXITED(stat_val) (((stat_val) & 255) == 0)
#  endif
# endif

#endif // !USE_SYSTEM

#ifdef HAVE_STROPTS_H
# ifdef sinix
#  define buf_T __system_buf_t__
# endif
# include <stropts.h>
# ifdef sinix
#  undef buf_T
# endif
#endif

#ifdef HAVE_STRING_H
# include <string.h>
#endif

#ifdef HAVE_SYS_STREAM_H
# include <sys/stream.h>
#endif

#ifdef HAVE_SYS_UTSNAME_H
# include <sys/utsname.h>
#endif

#ifdef HAVE_SYS_SYSTEMINFO_H
// <sys/systeminfo.h> uses SYS_NMLN but it may not be defined (CrayT3E).
# ifndef SYS_NMLN
#  define SYS_NMLN 32
# endif

# include <sys/systeminfo.h>	// for sysinfo
#endif

/*
 * We use termios.h if both termios.h and termio.h are available.
 * Termios is supposed to be a superset of termio.h.  Don't include them both,
 * it may give problems on some systems (e.g. hpux).
 * I don't understand why we don't want termios.h for apollo.
 */
#if defined(HAVE_TERMIOS_H) && !defined(apollo)
#  include <termios.h>
#else
# ifdef HAVE_TERMIO_H
#  include <termio.h>
# else
#  ifdef HAVE_SGTTY_H
#   include <sgtty.h>
#  endif
# endif
#endif

#ifdef HAVE_SYS_PTEM_H
# include <sys/ptem.h>	// must be after termios.h for Sinix
# ifndef _IO_PTEM_H	// For UnixWare that should check for _IO_PT_PTEM_H
#  define _IO_PTEM_H
# endif
#endif

// shared library access
#if defined(HAVE_DLFCN_H) && defined(USE_DLOPEN)
# if defined(__MVS__) && !defined (__SUSV3)
    // needed to define RTLD_LAZY (Anthony Giorgio)
#  define __SUSV3
# endif
# include <dlfcn.h>
#else
# if defined(HAVE_DL_H) && defined(HAVE_SHL_LOAD)
#  include <dl.h>
# endif
#endif
