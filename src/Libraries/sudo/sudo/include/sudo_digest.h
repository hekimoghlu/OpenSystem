/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#ifndef SUDO_DIGEST_H
#define SUDO_DIGEST_H

/* Digest types. */
#define SUDO_DIGEST_SHA224	0
#define SUDO_DIGEST_SHA256	1
#define SUDO_DIGEST_SHA384	2
#define SUDO_DIGEST_SHA512	3
#define SUDO_DIGEST_INVALID	4

struct sudo_digest;

/* Public functions. */
sudo_dso_public struct sudo_digest *sudo_digest_alloc_v1(int digest_type);
sudo_dso_public void sudo_digest_free_v1(struct sudo_digest *dig);
sudo_dso_public void sudo_digest_reset_v1(struct sudo_digest *dig);
sudo_dso_public int sudo_digest_getlen_v1(int digest_type);
sudo_dso_public void sudo_digest_update_v1(struct sudo_digest *dig, const void *data, size_t len);
sudo_dso_public void sudo_digest_final_v1(struct sudo_digest *dig, unsigned char *md);

#define sudo_digest_alloc(_a) sudo_digest_alloc_v1((_a))
#define sudo_digest_free(_a) sudo_digest_free_v1((_a))
#define sudo_digest_reset(_a) sudo_digest_reset_v1((_a))
#define sudo_digest_getlen(_a) sudo_digest_getlen_v1((_a))
#define sudo_digest_update(_a, _b, _c) sudo_digest_update_v1((_a), (_b), (_c))
#define sudo_digest_final(_a, _b) sudo_digest_final_v1((_a), (_b))

#endif /* SUDO_DIGEST_H */
