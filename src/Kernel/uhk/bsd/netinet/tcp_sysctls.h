/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef _NETINET_TCP_SYSCTLS_H_
#define _NETINET_TCP_SYSCTLS_H_

#include <sys/types.h>

extern int tcp_cubic_tcp_friendliness;
extern int tcp_cubic_fast_convergence;
extern int tcp_cubic_use_minrtt;
extern int tcp_cubic_minor_fixes;
extern int tcp_cubic_rfc_compliant;

extern int tcp_rack;

extern int target_qdelay;
extern int tcp_ledbat_allowed_increase;
extern int tcp_ledbat_tether_shift;
extern uint32_t bg_ss_fltsz;
extern int tcp_ledbat_plus_plus;

extern int tcp_rledbat;

extern int tcp_cc_debug;
extern int tcp_use_ledbat;
extern int tcp_use_newreno;

#endif /* _NETINET_TCP_SYSCTLS_H_ */
