/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#include <corecrypto/ccdh_gp.h>
#include <corecrypto/ccstubs.h>

static const ccdh_const_gp_t stub_gp = {
    .gp = NULL
};

ccdh_const_gp_t ccdh_gp_rfc5114_MODP_1024_160() {
	CC_STUB(stub_gp)
}

ccdh_const_gp_t ccdh_gp_rfc5114_MODP_2048_224() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc5114_MODP_2048_256() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group05() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group14() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group15() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group16() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group17() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc3526group18() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccsrp_gp_rfc5054_1024() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccsrp_gp_rfc5054_2048() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccsrp_gp_rfc5054_3072() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccsrp_gp_rfc5054_4096() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccsrp_gp_rfc5054_8192() {
	CC_STUB(stub_gp);
}

ccdh_const_gp_t ccdh_gp_rfc2409group02() {
	CC_STUB(stub_gp);
}
