/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
/* $OpenLDAP$ */
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#ifndef _AC_WAIT_H
#define _AC_WAIT_H

#include <sys/types.h>

#ifdef HAVE_SYS_WAIT_H
# include <sys/wait.h>
#endif

#define LDAP_HI(s)	(((s) >> 8) & 0377)
#define LDAP_LO(s)	((s) & 0377)

/* These should work on non-POSIX UNIX platforms,
	all bets on off on non-POSIX non-UNIX platforms... */
#ifndef WIFEXITED
# define WIFEXITED(s)	(LDAP_LO(s) == 0)
#endif
#ifndef WEXITSTATUS
# define WEXITSTATUS(s) LDAP_HI(s)
#endif
#ifndef WIFSIGNALED
# define WIFSIGNALED(s) (LDAP_LO(s) > 0 && LDAP_HI(s) == 0)
#endif
#ifndef WTERMSIG
# define WTERMSIG(s)	(LDAP_LO(s) & 0177)
#endif
#ifndef WIFSTOPPED
# define WIFSTOPPED(s)	(LDAP_LO(s) == 0177 && LDAP_HI(s) != 0)
#endif
#ifndef WSTOPSIG
# define WSTOPSIG(s)	LDAP_HI(s)
#endif

#ifdef WCONTINUED
# define WAIT_FLAGS ( WNOHANG | WUNTRACED | WCONTINUED )
#else
# define WAIT_FLAGS ( WNOHANG | WUNTRACED )
#endif

#endif /* _AC_WAIT_H */
