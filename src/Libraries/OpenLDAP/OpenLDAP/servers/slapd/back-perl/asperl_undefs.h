/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1999-2011 The OpenLDAP Foundation.
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

/* This file is probably obsolete.  If it is not,        */
/* #inclusion of it may have to be moved.  See ITS#2513. */

/* This file is necessary because both PERL headers */
/* and OpenLDAP define a number of macros without   */
/* checking wether they're already defined */

#ifndef ASPERL_UNDEFS_H
#define ASPERL_UNDEFS_H

/* ActiveState Win32 PERL port support */
/* set in ldap/include/portable.h */
#  ifdef HAVE_WIN32_ASPERL
/* The following macros are undefined to prevent */
/* redefinition in PERL headers*/
#    undef gid_t
#    undef uid_t
#    undef mode_t
#    undef caddr_t
#    undef WIN32_LEAN_AND_MEAN
#  endif
#endif

