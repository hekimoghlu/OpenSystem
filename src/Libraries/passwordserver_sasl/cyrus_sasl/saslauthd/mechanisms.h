/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#ifdef __GNUC__
#ident "$Id: mechanisms.h,v 1.9 2006/01/24 00:16:04 snsimon Exp $"
#endif

#ifndef _MECHANISMS_H
#define _MECHANISMS_H

#include "saslauthd.h"

/* PUBLIC DEPENDENCIES */
/* Authentication mechanism dispatch table definition */
typedef struct {
    char *name;				/* name of the mechanism */
    int (*initialize)(void);		/* initialization function */
    char *(*authenticate)(const char *, const char *,
			  const char *, const char *); /* authentication
							  function */
} authmech_t;

extern authmech_t mechanisms[];		/* array of supported auth mechs */
extern authmech_t *authmech;		/* auth mech daemon is using */
/* END PUBLIC DEPENDENCIES */

/*
 * Figure out which optional drivers we support.
 */
#ifndef AUTH_KRB5
# if defined(HAVE_KRB5_H) && defined(HAVE_GSSAPI)
#  define AUTH_KRB5
# endif
#endif

#ifndef AUTH_KRB4
# if defined(HAVE_KRB)
#  define AUTH_KRB4
# endif
#endif

#ifndef AUTH_DCE
# if defined(HAVE_USERSEC_H) && defined(HAVE_AUTHENTICATE)
#  define AUTH_DCE
# endif
#endif

#ifndef AUTH_SHADOW
# if defined(HAVE_GETSPNAM) || defined(HAVE_GETUSERPW)
#  define AUTH_SHADOW
# endif
#endif

#ifndef AUTH_SIA
# if defined(HAVE_SIA_VALIDATE_USER)
#  define AUTH_SIA
# endif
#endif

#ifndef AUTH_PAM
# ifdef HAVE_PAM
#  define AUTH_PAM
# endif
#endif

#ifndef AUTH_LDAP
# ifdef HAVE_LDAP
#  define AUTH_LDAP
# endif
#endif


#endif  /* _MECHANISMS_H */
