/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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

#include <name_code.h>

/* TLS library. */

#include <tls.h>

/* Application-specific. */

 /*
  * Numerical order of levels is critical (see tls.h):
  * 
  * - With "may" and higher, TLS is enabled.
  * 
  * - With "encrypt" and higher, TLS is required.
  * 
  * - With "fingerprint" and higher, the peer certificate must match.
  * 
  * - With "dane" and higher, the peer certificate must also be trusted,
  * possibly via TLSA RRs that make it its own authority.
  * 
  * The smtp(8) client will report trust failure in preference to reporting
  * failure to match, so we make "dane" larger than "fingerprint".
  */
static const NAME_CODE tls_level_table[] = {
    "none", TLS_LEV_NONE,
    "may", TLS_LEV_MAY,
    "encrypt", TLS_LEV_ENCRYPT,
    "fingerprint", TLS_LEV_FPRINT,
    "halfdane", TLS_LEV_HALF_DANE,	/* output only */
    "dane", TLS_LEV_DANE,
    "dane-only", TLS_LEV_DANE_ONLY,
    "verify", TLS_LEV_VERIFY,
    "secure", TLS_LEV_SECURE,
    0, TLS_LEV_INVALID,
};

int     tls_level_lookup(const char *name)
{
    int     level = name_code(tls_level_table, NAME_CODE_FLAG_NONE, name);

    return ((level != TLS_LEV_HALF_DANE) ? level : TLS_LEV_INVALID);
}

const char *str_tls_level(int level)
{
    return (str_name_code(tls_level_table, level));
}
