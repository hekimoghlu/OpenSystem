/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#ifndef _NETINET_MPTCP_SEQ_H_
#define _NETINET_MPTCP_SEQ_H_

/*
 * Use 64-bit modulo arithmetic for comparing
 * Data Sequence Numbers and Data ACKs. Implies
 * 2**63 space is available for sending data.
 */
#define MPTCP_SEQ_LT(a, b)      ((int64_t)((a) - (b)) < 0)
#define MPTCP_SEQ_LEQ(a, b)     ((int64_t)((a) - (b)) <= 0)
#define MPTCP_SEQ_GT(a, b)      ((int64_t)((a) - (b)) > 0)
#define MPTCP_SEQ_GEQ(a, b)     ((int64_t)((a) - (b)) >= 0)

#endif  /* _NETINET_MPTCP_SEQ_H_ */
