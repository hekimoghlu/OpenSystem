/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#include <corecrypto/ccnistkdf.h>
#include <corecrypto/ccstubs.h>
#include <stdio.h>

int ccnistkdf_ctr_hmac(const struct ccdigest_info *di,
                       size_t kdkLen, const void *kdk,
                       size_t labelLen, const void *label,
                       size_t contextLen, const void *context,
                       size_t dkLen, void *dk) {
	CC_STUB(0);
}

int ccnistkdf_ctr_hmac_fixed(const struct ccdigest_info *di,
                             size_t kdkLen, const void *kdk,
                             size_t fixedDataLen, const void *fixedData,
                             size_t dkLen, void *dk) {
	CC_STUB(0);
}

int ccnistkdf_fb_hmac(const struct ccdigest_info *di, int use_counter,
                      size_t kdkLen, const void *kdk,
                      size_t labelLen, const void *label,
                      size_t contextLen, const void *context,
                      size_t ivLen, const void *iv,
                      size_t dkLen, void *dk) {
	CC_STUB(0);
}

int ccnistkdf_fb_hmac_fixed(CC_UNUSED const struct ccdigest_info *di, int use_counter,
                            CC_UNUSED size_t kdkLen, CC_UNUSED const void *kdk,
                            CC_UNUSED size_t fixedDataLen, CC_UNUSED const void *fixedData,
                            CC_UNUSED size_t ivLen, CC_UNUSED const void *iv,
                            CC_UNUSED size_t dkLen, CC_UNUSED void *dk) {
	CC_STUB(0);
}

int ccnistkdf_dpi_hmac(const struct ccdigest_info *di,
                       size_t kdkLen, const void *kdk,
                       size_t labelLen, const void *label,
                       size_t contextLen, const void *context,
                       size_t dkLen, void *dk) {
	CC_STUB(0);
}

