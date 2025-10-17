/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
/* SYNOPSIS
 * mechanisms[] contains the NULL terminated list of supported
 * authentication drivers.
 * END SYNOPSIS */

#ifdef __GNUC__
#ident "$Id: mechanisms.c,v 1.9 2006/01/24 00:16:04 snsimon Exp $"
#endif

/* PUBLIC DEPENDENCIES */
#include "mechanisms.h"

#ifdef AUTH_DCE
# include "auth_dce.h"
#endif /* AUTH_DCE */
#ifdef AUTH_SHADOW
# include "auth_shadow.h"
#endif /* AUTH_SHADOW */
#ifdef AUTH_SIA
# include "auth_sia.h"
#endif /* AUTH_SIA */
#include "auth_krb4.h"
#include "auth_krb5.h"
#include "auth_getpwent.h"
#include "auth_sasldb.h"
#include "auth_rimap.h"
#ifdef AUTH_PAM
# include "auth_pam.h"
#endif
#ifdef AUTH_LDAP
#include "auth_ldap.h"
#endif
/* END PUBLIC DEPENDENCIES */

authmech_t mechanisms[] =
{
#ifdef AUTH_SASLDB
    {	"sasldb",	0,			auth_sasldb },
#endif /* AUTH_SASLDB */
#ifdef AUTH_DCE
    {	"dce",		0,			auth_dce },
#endif /* AUTH_DCE */
    {	"getpwent",	0,			auth_getpwent },
#ifdef AUTH_KRB4
    {	"kerberos4",	auth_krb4_init,		auth_krb4 },
#endif /* AUTH_KRB4 */
#ifdef AUTH_KRB5
    {	"kerberos5",	auth_krb5_init,		auth_krb5 },
#endif /* AUTH_KRB5 */
#ifdef AUTH_PAM
    {	"pam",		0,			auth_pam },
#endif /* AUTH_PAM */
    {	"rimap",	auth_rimap_init,	auth_rimap },
#ifdef AUTH_SHADOW
    {	"shadow",	0,			auth_shadow },
#endif /* AUTH_SHADOW */
#ifdef AUTH_SIA
    {   "sia",		0,			auth_sia },
#endif /* AUTH_SIA */
#ifdef AUTH_LDAP
    {   "ldap",		auth_ldap_init,		auth_ldap },
#endif /* AUTH_LDAP */
    {	0,		0,			0 }
};

