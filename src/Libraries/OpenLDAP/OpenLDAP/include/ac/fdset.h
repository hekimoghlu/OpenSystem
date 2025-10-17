/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

/*
 * This header is to be included by portable.h to ensure
 * tweaking of FD_SETSIZE is done early enough to be effective.
 */

#ifndef _AC_FDSET_H
#define _AC_FDSET_H

#if !defined( OPENLDAP_FD_SETSIZE ) && !defined( FD_SETSIZE )
#  define OPENLDAP_FD_SETSIZE 4096
#endif

#ifdef OPENLDAP_FD_SETSIZE
    /* assume installer desires to enlarge fd_set */
#  ifdef HAVE_BITS_TYPES_H
#    include <bits/types.h>
#  endif
#  ifdef __FD_SETSIZE
#    undef __FD_SETSIZE
#    define __FD_SETSIZE OPENLDAP_FD_SETSIZE
#  else
#    define FD_SETSIZE OPENLDAP_FD_SETSIZE
#  endif
#endif

#endif /* _AC_FDSET_H */
