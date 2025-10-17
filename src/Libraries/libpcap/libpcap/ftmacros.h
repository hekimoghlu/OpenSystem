/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#ifndef ftmacros_h
#define	ftmacros_h

/*
 * Define some feature test macros to make sure that everything we want
 * to be declared gets declared.
 *
 * On some UN*Xes we need to force strtok_r() to be declared.
 * We do *NOT* want to define _POSIX_C_SOURCE, as that tends
 * to make non-POSIX APIs that we use unavailable.
 * XXX - is there no portable way to say "please pollute the
 * namespace to the maximum extent possible"?
 */
#if defined(sun) || defined(__sun)
  #define __EXTENSIONS__

  /*
   * We also need to define _XPG4_2 in order to get
   * the Single UNIX Specification version of
   * recvmsg().
   */
  #define _XPG4_2
#elif defined(_hpux) || defined(hpux) || defined(__hpux)
  #define _REENTRANT

  /*
   * We need this to get the versions of socket functions that
   * use socklen_t.  Define it only if it's not already defined,
   * so we don't get redefiniton warnings.
   */
  #ifndef _XOPEN_SOURCE_EXTENDED
    #define _XOPEN_SOURCE_EXTENDED
  #endif

  /*
   * XXX - the list of PA-RISC options for GCC makes it sound as if
   * building code that uses a particular vintage of UNIX API/ABI
   * is complicated:
   *
   *    https://gcc.gnu.org/onlinedocs/gcc/HPPA-Options.html
   *
   * See the description of the -munix flag.
   *
   * We probably want libpcap to work with programs built for any
   * UN*X standard.  I'm not sure whether that's possible and, if
   * it is, what sort of stuff it'd have to do.
   *
   * It might also be a requirement that we build with a special
   * flag to allow the library to be used with threaded code, at
   * least with HP's C compiler; hopefully doing so won't make it
   * *not* work with *un*-threaded code.
   */
#else
  /*
   * Turn on _GNU_SOURCE to get everything GNU libc has to offer,
   * including asprintf(), if we're using GNU libc.
   *
   * Unfortunately, one thing it has to offer is a strerror_r()
   * that's not POSIX-compliant, but we deal with that in
   * pcap_fmt_errmsg_for_errno().
   *
   * We don't limit this to, for example, Linux and Cygwin, because
   * this might, for example, be GNU/HURD or one of Debian's kFreeBSD
   * OSes ("GNU/FreeBSD").
   */
  #define _GNU_SOURCE

  /*
   * We turn on both _DEFAULT_SOURCE and _BSD_SOURCE to try to get
   * the BSD u_XXX types, such as u_int and u_short, defined.  We
   * define _DEFAULT_SOURCE first, so that newer versions of GNU libc
   * don't whine about _BSD_SOURCE being deprecated; we still have
   * to define _BSD_SOURCE to handle older versions of GNU libc that
   * don't support _DEFAULT_SOURCE.
   *
   * But, if it's already defined, don't define it, so that we don't
   * get a warning of it being redefined if it's defined as, for
   * example, 1.
   */
  #ifndef _DEFAULT_SOURCE
    #define _DEFAULT_SOURCE
  #endif
  /* Avoid redefining _BSD_SOURCE if it's already defined as for ex. 1 */
  #ifndef _BSD_SOURCE
    #define _BSD_SOURCE
  #endif
#endif

#endif
