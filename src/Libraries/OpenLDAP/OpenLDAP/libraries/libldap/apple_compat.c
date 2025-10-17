/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

#include <unistd.h>
#include <ldap.h>

int ldap_get_lderrno(LDAP *ld, char **m, char **s)
{
	int rc, lderrno;

	rc = ldap_get_option(ld, LDAP_OPT_ERROR_NUMBER, &lderrno);
	if (rc != LDAP_SUCCESS)
		return rc;

	if (s != NULL)
	{
		rc = ldap_get_option(ld, LDAP_OPT_ERROR_STRING, s);
		if (rc != LDAP_SUCCESS)
			return rc;
	}

	if (s != NULL)
	{
		rc = ldap_get_option(ld, LDAP_OPT_MATCHED_DN, m);
		if (rc != LDAP_SUCCESS)
			return rc;
	}

	return lderrno;
}

int ldap_set_lderrno(LDAP *ld, int lderrno, const char *m, const char *s)
{
	int rc;

	rc = ldap_set_option(ld, LDAP_OPT_ERROR_NUMBER, &lderrno);
	if (rc != LDAP_SUCCESS)
		return rc;

	if (s != NULL)
	{
		rc = ldap_set_option(ld, LDAP_OPT_ERROR_STRING, s);
		if (rc != LDAP_SUCCESS)
			return rc;
	}

	if (m != NULL)
	{
		rc = ldap_set_option(ld, LDAP_OPT_MATCHED_DN, m);
		if (rc != LDAP_SUCCESS)
			return rc;
	}

	return LDAP_SUCCESS;
}

int ldap_version(void *ver)
{
	return LDAP_API_VERSION;
}
