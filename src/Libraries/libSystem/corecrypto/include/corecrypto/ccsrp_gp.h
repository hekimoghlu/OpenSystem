/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#ifndef corecrypto_ccsrp_gp_h
#define corecrypto_ccsrp_gp_h

#include <corecrypto/ccdh.h>

ccdh_const_gp_t ccsrp_gp_rfc5054_1024(void);
ccdh_const_gp_t ccsrp_gp_rfc5054_2048(void);
ccdh_const_gp_t ccsrp_gp_rfc5054_3072(void);
ccdh_const_gp_t ccsrp_gp_rfc5054_4096(void);
ccdh_const_gp_t ccsrp_gp_rfc5054_8192(void);

#endif
