/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
 * tpOcspVerify.h - top-level OCSP verification
 */
 
#ifndef	_TP_OCSP_VERIFY_H_
#define _TP_OCSP_VERIFY_H_

#include "tpCrlVerify.h"

extern "C" {

/* 
 * The sole and deceptively simple looking public interface to this module. 
 * It's pretty heavyweight; expect to spend millions or billions of cycles
 * here before returning. 
 */
CSSM_RETURN tpVerifyCertGroupWithOCSP(
	TPVerifyContext	&tpVerifyContext,
	TPCertGroup 	&certGroup);		// to be verified 
	
}

#endif	/* _TP_OCSP_VERIFY_H_ */

