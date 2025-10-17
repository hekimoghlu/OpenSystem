/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#ifndef GSSKRB5_CRYPTO_H
#define GSSKRB5_CRYPTO_H 1

struct gss_msg_order;

struct gsskrb5_crypto {
    krb5_crypto crypto;
    uint32_t flags;
#define GK5C_ACCEPTOR        (1 << 0)
#define GK5C_ACCEPTOR_SUBKEY (1 << 2)
#define GK5C_DCE_STYLE	     (1 << 9)
    uint32_t seqnumlo;
    uint32_t seqnumhi;
    struct gss_msg_order *order;
};

#define GK5C_IS_DCE_STYLE(ctx) ((ctx)->flags & GK5C_DCE_STYLE)

#endif /* GSSKRB5_CRYPTO_H */
