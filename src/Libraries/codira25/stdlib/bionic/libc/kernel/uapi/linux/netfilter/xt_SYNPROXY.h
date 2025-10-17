/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#ifndef _XT_SYNPROXY_H
#define _XT_SYNPROXY_H
#include <linux/netfilter/nf_synproxy.h>
#define XT_SYNPROXY_OPT_MSS NF_SYNPROXY_OPT_MSS
#define XT_SYNPROXY_OPT_WSCALE NF_SYNPROXY_OPT_WSCALE
#define XT_SYNPROXY_OPT_SACK_PERM NF_SYNPROXY_OPT_SACK_PERM
#define XT_SYNPROXY_OPT_TIMESTAMP NF_SYNPROXY_OPT_TIMESTAMP
#define XT_SYNPROXY_OPT_ECN NF_SYNPROXY_OPT_ECN
#define xt_synproxy_info nf_synproxy_info
#endif
