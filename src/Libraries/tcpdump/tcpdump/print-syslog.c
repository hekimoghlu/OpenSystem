/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
/* \summary: Syslog protocol printer */
/* specification: RFC 3164 (not RFC 5424) */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"


/*
 * tokenlists and #defines taken from Ethereal - Network traffic analyzer
 * by Gerald Combs <gerald@ethereal.com>
 */

#define SYSLOG_SEVERITY_MASK 0x0007  /* 0000 0000 0000 0111 */
#define SYSLOG_FACILITY_MASK 0x03f8  /* 0000 0011 1111 1000 */
#define SYSLOG_MAX_DIGITS 3 /* The maximum number of priority digits to read in. */

static const struct tok syslog_severity_values[] = {
  { 0,      "emergency" },
  { 1,      "alert" },
  { 2,      "critical" },
  { 3,      "error" },
  { 4,      "warning" },
  { 5,      "notice" },
  { 6,      "info" },
  { 7,      "debug" },
  { 0, NULL },
};

static const struct tok syslog_facility_values[] = {
  { 0,     "kernel" },
  { 1,     "user" },
  { 2,     "mail" },
  { 3,     "daemon" },
  { 4,     "auth" },
  { 5,     "syslog" },
  { 6,     "lpr" },
  { 7,     "news" },
  { 8,     "uucp" },
  { 9,     "cron" },
  { 10,    "authpriv" },
  { 11,    "ftp" },
  { 12,    "ntp" },
  { 13,    "security" },
  { 14,    "console" },
  { 15,    "cron" },
  { 16,    "local0" },
  { 17,    "local1" },
  { 18,    "local2" },
  { 19,    "local3" },
  { 20,    "local4" },
  { 21,    "local5" },
  { 22,    "local6" },
  { 23,    "local7" },
  { 0, NULL },
};

void
syslog_print(netdissect_options *ndo,
             const u_char *pptr, u_int len)
{
    uint16_t msg_off = 0;
    uint16_t pri = 0;
    uint16_t facility,severity;

    ndo->ndo_protocol = "syslog";
    /* extract decimal figures that are
     * encapsulated within < > tags
     * based on this decimal figure extract the
     * severity and facility values
     */

    if (GET_U_1(pptr) != '<')
        goto invalid;
    msg_off++;

    while (msg_off <= SYSLOG_MAX_DIGITS &&
           GET_U_1(pptr + msg_off) >= '0' &&
           GET_U_1(pptr + msg_off) <= '9') {
        pri = pri * 10 + (GET_U_1(pptr + msg_off) - '0');
        msg_off++;
    }

    if (GET_U_1(pptr + msg_off) != '>')
        goto invalid;
    msg_off++;

    facility = (pri & SYSLOG_FACILITY_MASK) >> 3;
    severity = pri & SYSLOG_SEVERITY_MASK;

    if (ndo->ndo_vflag < 1 )
    {
        ND_PRINT("SYSLOG %s.%s, length: %u",
               tok2str(syslog_facility_values, "unknown (%u)", facility),
               tok2str(syslog_severity_values, "unknown (%u)", severity),
               len);
        return;
    }

    ND_PRINT("SYSLOG, length: %u\n\tFacility %s (%u), Severity %s (%u)\n\tMsg: ",
           len,
           tok2str(syslog_facility_values, "unknown (%u)", facility),
           facility,
           tok2str(syslog_severity_values, "unknown (%u)", severity),
           severity);

    /* print the syslog text in verbose mode */
    /*
     * RFC 3164 Section 4.1.3: "There is no ending delimiter to this part.
     * The MSG part of the syslog packet MUST contain visible (printing)
     * characters."
     *
     * RFC 5424 Section 8.2: "This document does not impose any mandatory
     * restrictions on the MSG or PARAM-VALUE content.  As such, they MAY
     * contain control characters, including the NUL character."
     *
     * Hence, to aid in protocol debugging, print the full MSG without
     * beautification to make it clear what was transmitted on the wire.
     */
    if (len > msg_off)
        (void)nd_printn(ndo, pptr + msg_off, len - msg_off, NULL);

    if (ndo->ndo_vflag > 1)
        print_unknown_data(ndo, pptr, "\n\t", len);
    return;

invalid:
    nd_print_invalid(ndo);
}
