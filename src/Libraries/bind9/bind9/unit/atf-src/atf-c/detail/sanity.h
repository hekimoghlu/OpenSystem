/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#if !defined(ATF_C_SANITY_H)
#define ATF_C_SANITY_H

void atf_sanity_inv(const char *, int, const char *);
void atf_sanity_pre(const char *, int, const char *);
void atf_sanity_post(const char *, int, const char *);

#if !defined(NDEBUG)

#define INV(x) \
    do { \
        if (!(x)) \
            atf_sanity_inv(__FILE__, __LINE__, #x); \
    } while (0)
#define PRE(x) \
    do { \
        if (!(x)) \
            atf_sanity_pre(__FILE__, __LINE__, #x); \
    } while (0)
#define POST(x) \
    do { \
        if (!(x)) \
            atf_sanity_post(__FILE__, __LINE__, #x); \
    } while (0)

#else /* defined(NDEBUG) */

#define INV(x) \
    do { \
	(void)(x); \
    } while (0)

#define PRE(x) \
    do { \
	(void)(x); \
    } while (0)

#define POST(x) \
    do { \
	(void)(x); \
    } while (0)

#endif /* !defined(NDEBUG) */

#define UNREACHABLE INV(0)

#endif /* ATF_C_SANITY_H */
