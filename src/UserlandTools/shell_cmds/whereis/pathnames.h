/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
/* Where to look for libexec */
#define PATH_LIBEXEC "/usr/libexec"

#ifndef __APPLE__
/* Where to look for sources. */
#define PATH_SOURCES					\
"/usr/src/bin:/usr/src/usr.bin:/usr/src/sbin:"		\
"/usr/src/usr.sbin:/usr/src/libexec:"			\
"/usr/src/gnu/bin:/usr/src/gnu/usr.bin:"		\
"/usr/src/gnu/sbin:/usr/src/gnu/usr.sbin:"		\
"/usr/src/contrib:"					\
"/usr/src/secure/bin:/usr/src/secure/usr.bin:"		\
"/usr/src/secure/sbin:/usr/src/secure/usr.sbin:"	\
"/usr/src/secure/libexec:/usr/src/crypto:"		\
"/usr/src/games"

/* Each subdirectory of PATH_PORTS will be appended to PATH_SOURCES. */
#define PATH_PORTS "/usr/ports"
#endif

/* How to query the current manpath. */
#define MANPATHCMD "manpath -q"

/* How to obtain the location of manpages, and how to match this result. */
#define MANWHEREISCMD "man -S1:8:6 -w %s 2>/dev/null"
#define MANWHEREISALLCMD "man -a -w %s 2>/dev/null"
#define MANWHEREISMATCH "^.* [(]source: (.*)[)]$"

/* Command used to locate sources that have not been found yet. */
#ifndef __APPLE__
#define LOCATECMD "locate '*'/%s 2>/dev/null"
#endif
