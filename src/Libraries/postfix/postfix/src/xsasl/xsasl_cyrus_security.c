/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <name_mask.h>

/* Application-specific. */

#include <xsasl_cyrus_common.h>

#if defined(USE_SASL_AUTH) && defined(USE_CYRUS_SASL)

#include <sasl.h>

 /*
  * SASL Security options.
  */
static const NAME_MASK xsasl_cyrus_sec_mask[] = {
    "noplaintext", SASL_SEC_NOPLAINTEXT,
    "noactive", SASL_SEC_NOACTIVE,
    "nodictionary", SASL_SEC_NODICTIONARY,
#ifdef SASL_SEC_FORWARD_SECRECY
    "forward_secrecy", SASL_SEC_FORWARD_SECRECY,
#endif
    "noanonymous", SASL_SEC_NOANONYMOUS,
#if SASL_VERSION_MAJOR >= 2
    "mutual_auth", SASL_SEC_MUTUAL_AUTH,
#endif
    0,
};

/* xsasl_cyrus_security - parse security options */

int     xsasl_cyrus_security_parse_opts(const char *sasl_opts_val)
{
    return (name_mask_opt("SASL security options", xsasl_cyrus_sec_mask,
			  sasl_opts_val, NAME_MASK_RETURN));
}

#endif
