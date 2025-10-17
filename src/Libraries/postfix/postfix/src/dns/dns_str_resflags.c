/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <netinet/in.h>
#include <arpa/nameser.h>
#include <resolv.h>

 /*
  * Utility library.
  */
#include <name_mask.h>

 /*
  * DNS library.
  */
#include <dns.h>

 /*
  * Application-specific.
  */

 /*
  * This list overlaps with dns_res_opt_masks[] in smtp.c, but there we
  * permit only a small subset of all possible flags.
  */
static const LONG_NAME_MASK resflag_table[] = {
    "RES_INIT", RES_INIT,
    "RES_DEBUG", RES_DEBUG,
    "RES_AAONLY", RES_AAONLY,
    "RES_USEVC", RES_USEVC,
    "RES_PRIMARY", RES_PRIMARY,
    "RES_IGNTC", RES_IGNTC,
    "RES_RECURSE", RES_RECURSE,
    "RES_DEFNAMES", RES_DEFNAMES,
    "RES_STAYOPEN", RES_STAYOPEN,
    "RES_DNSRCH", RES_DNSRCH,
    "RES_INSECURE1", RES_INSECURE1,
    "RES_INSECURE2", RES_INSECURE2,
    "RES_NOALIASES", RES_NOALIASES,
    "RES_USE_INET6", RES_USE_INET6,
#ifdef RES_ROTATE
    "RES_ROTATE", RES_ROTATE,
#endif
#ifdef RES_NOCHECKNAME
    "RES_NOCHECKNAME", RES_NOCHECKNAME,
#endif
    "RES_USE_EDNS0", RES_USE_EDNS0,
    "RES_USE_DNSSEC", RES_USE_DNSSEC,
#ifdef RES_KEEPTSIG
    "RES_KEEPTSIG", RES_KEEPTSIG,
#endif
#ifdef RES_BLAST
    "RES_BLAST", RES_BLAST,
#endif
#ifdef RES_USEBSTRING
    "RES_USEBSTRING", RES_USEBSTRING,
#endif
#ifdef RES_NSID
    "RES_NSID", RES_NSID,
#endif
#ifdef RES_NOIP6DOTINT
    "RES_NOIP6DOTINT", RES_NOIP6DOTINT,
#endif
#ifdef RES_USE_DNAME
    "RES_USE_DNAME", RES_USE_DNAME,
#endif
#ifdef RES_NO_NIBBLE2
    "RES_NO_NIBBLE2", RES_NO_NIBBLE2,
#endif
#ifdef RES_SNGLKUP
    "RES_SNGLKUP", RES_SNGLKUP,
#endif
#ifdef RES_SNGLKUPREOP
    "RES_SNGLKUPREOP", RES_SNGLKUPREOP,
#endif
#ifdef RES_NOTLDQUERY
    "RES_NOTLDQUERY", RES_NOTLDQUERY,
#endif
    0,
};

/* dns_str_resflags - convert RES_* resolver flags to printable form */

const char *dns_str_resflags(unsigned long mask)
{
    static VSTRING *buf;

    if (buf == 0)
	buf = vstring_alloc(20);
    return (str_long_name_mask_opt(buf, "dsns_str_resflags", resflag_table,
				   mask, NAME_MASK_NUMBER | NAME_MASK_PIPE));
}
