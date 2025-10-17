/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

#ifndef CC_PRIVATE_CCEC_EXTRA_H
#define CC_PRIVATE_CCEC_EXTRA_H

#define ccec_cp_ccn_size(x) ((8 * ccec_cp_n(x)) + 3)
#define ccec_cp_size(x) (sizeof(struct cczp) + ccec_cp_ccn_size(x))

#define ccec_cp_p(x)  (x.prime->ccn)
#define ccec_cp_pr(x) (x.prime->ccn + 1 * ccec_cp_n(x))
#define ccec_cp_a(x)  (x.prime->ccn + 2 * ccec_cp_n(x) + 1)
#define ccec_cp_b(x)  (x.prime->ccn + 3 * ccec_cp_n(x) + 1)
#define ccec_cp_x(x)  (x.prime->ccn + 4 * ccec_cp_n(x) + 1)
#define ccec_cp_y(x)  (x.prime->ccn + 5 * ccec_cp_n(x) + 1)
#define ccec_cp_o(x)  (x.prime->ccn + 6 * ccec_cp_n(x) + 1)
#define ccec_cp_or(x) (x.prime->ccn + 7 * ccec_cp_n(x) + 1)
#define ccec_cp_h(x)  (x.prime->ccn + 8 * ccec_cp_n(x) + 2)

#endif // CC_PRIVATE_CCEC_EXTRA_H
