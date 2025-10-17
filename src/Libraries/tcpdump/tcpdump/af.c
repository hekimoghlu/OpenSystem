/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"
#include "netdissect.h"
#include "af.h"

const struct tok af_values[] = {
    { 0,                      "Reserved"},
    { AFNUM_INET,             "IPv4"},
    { AFNUM_INET6,            "IPv6"},
    { AFNUM_NSAP,             "NSAP"},
    { AFNUM_HDLC,             "HDLC"},
    { AFNUM_BBN1822,          "BBN 1822"},
    { AFNUM_802,              "802"},
    { AFNUM_E163,             "E.163"},
    { AFNUM_E164,             "E.164"},
    { AFNUM_F69,              "F.69"},
    { AFNUM_X121,             "X.121"},
    { AFNUM_IPX,              "Novell IPX"},
    { AFNUM_ATALK,            "Appletalk"},
    { AFNUM_DECNET,           "Decnet IV"},
    { AFNUM_BANYAN,           "Banyan Vines"},
    { AFNUM_E164NSAP,         "E.164 with NSAP subaddress"},
    { AFNUM_L2VPN,            "Layer-2 VPN"},
    { AFNUM_VPLS,             "VPLS"},
    { 0, NULL},
};

const struct tok bsd_af_values[] = {
    { BSD_AFNUM_INET, "IPv4" },
    { BSD_AFNUM_NS, "NS" },
    { BSD_AFNUM_ISO, "ISO" },
    { BSD_AFNUM_APPLETALK, "Appletalk" },
    { BSD_AFNUM_IPX, "IPX" },
    { BSD_AFNUM_INET6_BSD, "IPv6" },
    { BSD_AFNUM_INET6_FREEBSD, "IPv6" },
    { BSD_AFNUM_INET6_DARWIN, "IPv6" },
    { 0, NULL}
};
